import yaml
from argparse import Namespace
from pathlib import Path

from tqdm.auto import tqdm
from pprint import pprint

import torch
from transformers import AutoTokenizer
import transformer_lens
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

from .circuit_lm import CircuitTransformer
from .data import setup_task
from .evaluation import compute_complete_loss, compute_faith_loss
from .utils import schedule_epoch_lambda

class DiscoGP:
    def __init__(self, args, configs_path):
        self.args = args
        self.configs = self.load_configs(configs_path)

        self.device = torch.device('cuda')
        self.setup_model()
        self.setup_task()

    def load_configs(self, configs_path):
        with open(configs_path) as configs_file:
            configs_dict = yaml.safe_load(configs_file)

            for name, value in configs_dict.items():
                if name.endswith('path'):
                    configs_dict[name] = Path(value)

            configs = Namespace(**configs_dict)
            configs.weight_hparams = Namespace(**configs.weight_hparams)
            configs.edge_hparams = Namespace(**configs.edge_hparams)
        return configs

    def setup_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = CircuitTransformer.from_pretrained(self.args.model_name,
            debug=False,
            gs_temp_weight=self.configs.weight_hparams.gs_temp,
            gs_temp_edge=self.configs.edge_hparams.gs_temp,
            use_weight_masks=self.args.use_weight_masks,
            use_edge_masks=self.args.use_edge_masks,
        ).to(self.device)

    def setup_task(self):
        self.dls = setup_task(self)

    @torch.no_grad
    def evaluate(self, dl=None, reverse=False):
        if dl is None:
            dl = self.dls.eval

        self.model.eval()

        self.model.turn_on_weight_masks(deterministic=True, reverse=reverse)
        self.model.turn_on_edge_masks(deterministic=True, reverse=reverse)

        if self.args.use_weight_masks:
            _, _, weight_density = self.model.get_weight_density()
        else:
            weight_density = 'na'
        if self.args.use_edge_masks:
            _, _, edge_density = self.model.get_edge_density()
        else:
            edge_density = 'na'

        if not self.args.use_weight_masks:
            self.model.turn_off_weight_masks()
        if not self.args.use_edge_masks:
            self.model.turn_off_edge_masks()

        total = len(dl.dataset)
        correct = 0
        kls = []
        faith_losses = []

        for batch_inputs in tqdm(dl):

            batch_logits_masked = self.model(batch_inputs['input_ids'].to(self.device))[0]
            eval_results = self.compute_loss(batch_logits_masked, batch_inputs)

            correct += eval_results['n_correct']
            # kls.append(eval_results['kl_div'].cpu())
            faith_losses.append(eval_results['faith'])

        self.model.turn_off_weight_masks()
        self.model.turn_off_edge_masks()

        acc = correct / total
        
        results = {
            'acc': acc,
            # 'kl': torch.stack(kls).mean().item(),
            'faith_loss': torch.stack(faith_losses).mean().item(),
            'weight_density': weight_density,
            'edge_density': edge_density,
            'n_correct': correct,
            'total': total
        }

        pprint(results)
        return results


    def compute_loss(self, batch_logits_masked, batch_inputs):
        results = {}

        faith_results = compute_faith_loss(batch_logits_masked, batch_inputs)
        results.update(faith_results)

        return faith_results

    def search_circuit(self):
        if self.args.use_weight_masks:
            self.run_prune('w')

        if self.args.use_edge_masks:
            self.run_prune('e')

    def run_prune(self, mode):

        if mode == 'w':
            # weight pruning
            mask_logits_dict = self.model.mask_logits_dict_weight
            hparams = self.configs.weight_hparams
        elif mode == 'e':
            mask_logits_dict = self.model.mask_logits_dict_edge
            hparams = self.configs.edge_hparams

        mask_logits = [mask for _, mask in mask_logits_dict.items()]
        optimizer = torch.optim.AdamW(mask_logits, lr=hparams.lr)

        if mode == 'w':
            self.model.turn_off_edge_masks()
        elif mode == 'e':
            self.model.turn_on_weight_masks(deterministic=True)

        for epoch in tqdm(range(hparams.train_epochs)):
            lambda_sparse = schedule_epoch_lambda(
                epoch, 
                lambda_0=hparams.lambda_sparse_init,
                max_times=hparams.max_times_lambda_sparse, 
                min_times=hparams.min_times_lambda_sparse, 
                n_epoch_warmup=hparams.n_epoch_warmup_lambda_sparse,
                n_epoch_cooldown=hparams.n_epoch_cooldown_lambda_sparse,
            )
            lambda_complete = schedule_epoch_lambda(epoch, hparams.lambda_complete_init)

            for batch_inputs in tqdm(self.dls.train):

                # weight pruning
                if mode == 'w':
                    self.model.turn_on_weight_masks(deterministic=False, reverse=False)
                    sparse_loss = self.model.weight_sparseness_loss()
                elif mode == 'e':
                    self.model.turn_on_edge_masks(deterministic=False, reverse=False)
                    sparse_loss = self.model.edge_sparseness_loss()

                batch_logits_masked = self.model(batch_inputs['input_ids'].to(self.device))[0] # (B, seq_len, vocab_size)
                eval_results = compute_faith_loss(batch_logits_masked, batch_inputs)
                faith_loss = eval_results['faith']

                if mode == 'e' and lambda_complete > 0:
                    self.model.turn_on_edge_masks(deterministic=False, reverse=True)
                    batch_logits = self.model(batch_inputs['input_ids'].to(self.device))[0] 
                    complete_loss, _ = compute_complete_loss(batch_logits, batch_inputs) 
                else:
                    complete_loss = 0.

                if mode == 'w':
                    loss = faith_loss + sparse_loss * lambda_sparse
                elif mode == 'e':
                    loss = faith_loss + sparse_loss * lambda_sparse + complete_loss * lambda_complete

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if mode == 'w':
                    self.model.turn_off_weight_masks()

                if mode == 'w' and lambda_complete > 0:
                    self.model.turn_on_weight_masks(deterministic=False, reverse=True)
                    batch_logits = self.model(batch_inputs['input_ids'].to(self.device))[0]
                    complete_loss, _ = compute_complete_loss(batch_logits, batch_inputs)
                    loss = complete_loss * lambda_complete
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    self.model.turn_off_weight_masks()

            use_weight_mask = mode == 'w'
            use_edge_mask = mode == 'e'
            
            eval_results_faith = self.evaluate(dl=self.dls.eval, reverse=False)
            eval_results_complement = self.evaluate(dl=self.dls.eval, reverse=True)
