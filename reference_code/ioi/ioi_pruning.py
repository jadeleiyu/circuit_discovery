from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import transformer_lens
from torch.optim import AdamW
from os.path import join
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import argparse

import sys
sys.path.append('/home/leiyu/projects/def-yangxu/leiyu/circuit-discovery')
from dmc.circuit_gpt import *
from ioi_dataset import *
from ioi_utils import *


def main(args):

    # path that stores gpt-small weights and gpt tokenizer
    model_path = join(args.model_dir, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # load IOI data
    # Note that there are overlaps between train and test sets, due to the way IOIDataset is constructed (randomly sample N items)
    ioi_prompts = pickle.load(open(join(args.data_dir, f'ioi_prompts_{args.ds_idx}.p'), 'rb'))
    full_model_target_log_probs = torch.load(f'full_model_results/target_log_probs_{args.ds_idx}.pt')
    full_model_pred_labels = torch.load(f'full_model_results/pred_labels_{args.ds_idx}.pt')
    
    data_dict = prepare_ioi_data_for_clm(ioi_prompts, full_model_target_log_probs, full_model_pred_labels)
    ds = IOICircuitDataset(data_dict)
    n_train = int(0.8*len(ds))
    n_test = len(ds) - n_train
    ds_train, ds_test = torch.utils.data.random_split(ds, [n_train, n_test])
    
    train_dl = DataLoader(
        ds_train,
        batch_size=args.batch_size
    )
    eval_dl = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False
    )

    # circuit_gpt initialization
    device = torch.device('cuda')    
    gpt_weights = torch.load(join(model_path, 'model_weights.pt')) 
    circuit_gpt_config = CircuitGPTConfig(
        debug=False,
        gs_temp_weight=args.gs_temp_weight,
        gs_temp_edge=args.gs_temp_edge,
        use_weight_masks=args.use_weight_masks
    )
    circuit_gpt = CircuitGPT(circuit_gpt_config)
    circuit_gpt.load_pretrained_weight(gpt_weights)
    
    # load pretrained mask logits if necessary
    if args.resume_epoch_w > 0:
        weight_mask_logits = torch.load(join(args.results_dir, f'mask_logits_dict_weight_ioi_{args.ds_idx}_weight_{args.resume_epoch_w}_edge_0.pt'))
        circuit_gpt.load_pretrained_weight_mask(weight_mask_logits)
    # if args.resume_epoch_e > 0:
    #     edge_mask_logits = torch.load(join(mask_path), f'ioi/edge_mask_logits_{args.ds_idx}_{resume_epoch_e}.pt')
    #     circuit_gpt.load_pretrained_edge_mask(edge_mask_logits)
        
    # move circuit_gpt to gpu after loading all weights and masks
    circuit_gpt.to(device);

    print(f'task name: ioi')
    print(f'dataset idx: {args.ds_idx}')
    print(f'prune weight: {args.prune_weight}')
    print(f'use weight masks: {args.use_weight_masks}')
    print(f'prune edge: {args.prune_edge}')
    print(f'optimize weight completeness: {args.lambda_complete_weight_init > 0}')
    print(f'optimize edge completeness: {args.lambda_complete_edge_init > 0}')

    # evaluation before any weight/edge pruning
    eval_results_full_model = eval_model(circuit_gpt, eval_dl, tokenizer, device, 
        use_weight_mask=False, use_edge_mask=False, reverse=False
    )
    print(
        f"Epoch 0. mean pruned model eval accuracy: {eval_results_full_model['acc']:.2f}," + 
        f"mean eval kl-div: {eval_results_full_model['kl']:.4f}," + 
        f"weight density: {eval_results_full_model['weight_density']:.4f}," + 
        f"edge density: {eval_results_full_model['edge_density']:.4f}"
    )

    if args.prune_weight:
        n_epoch_w = args.train_epochs_weight
        weight_logits = [mask for _, mask in circuit_gpt.mask_logits_dict_weight.items()]
        optim_weight = torch.optim.AdamW(weight_logits, lr=args.lr_weight)
        circuit_gpt.turn_off_edge_masks()        
        for epoch in tqdm(range(args.train_epochs_weight)):
            
            lambda_sparse_weight = schedule_epoch_lambda(
                epoch, args.lambda_sparse_weight_init, 
                max_times=args.max_times_lambda_sparse_weight, 
                min_times=args.min_times_lambda_sparse_weight, 
                n_epoch_warmup=args.n_epoch_warmup_lambda_sparse_weight,
                n_epoch_cooldown=args.n_epoch_cooldown_lambda_sparse_weight
            )
            lambda_complete_weight = schedule_epoch_lambda(epoch, args.lambda_complete_weight_init)
            # lambda_complete_weight = 0
            for batch in train_dl:
                batch_inputs = prepare_batch_inputs(batch, tokenizer)
                    
                # weight pruning
                circuit_gpt.turn_on_weight_masks(deterministic=False, reverse=False)
                sparse_loss_weight = circuit_gpt.weight_sparseness_loss()
                batch_logits_masked = circuit_gpt(batch_inputs['input_ids'].to(device))[0] # (B, seq_len, vocab_size)                
                faith_loss_weight, _, _ = compute_faith_loss(batch_logits_masked, batch_inputs)

                # first backprop the sparseness and the faithful loss before computing the completeness loss
                # since the completeness loss requires changing model weights in place
                loss_weight = faith_loss_weight + sparse_loss_weight * lambda_sparse_weight
                # loss_weight = faith_loss_weight
                loss_weight.backward()
                optim_weight.step()
                optim_weight.zero_grad()
                circuit_gpt.turn_off_weight_masks()  
                # torch.cuda.empty_cache()

                if lambda_complete_weight > 0:
                    circuit_gpt.turn_on_weight_masks(deterministic=False, reverse=True)
                    batch_logits = circuit_gpt(batch_inputs['input_ids'].to(device))[0] 
                    complete_loss_weight, _ = compute_complete_loss(batch_logits, batch_inputs)         
                    loss_weight = complete_loss_weight * lambda_complete_weight
                    loss_weight.backward()                    
                    optim_weight.step()
                    optim_weight.zero_grad()    
                    circuit_gpt.turn_off_weight_masks()  
                    # torch.cuda.empty_cache()
        
            eval_results_faith = eval_model(
                circuit_gpt, eval_dl, tokenizer, device, use_weight_mask=True, use_edge_mask=False, reverse=False
            )
            eval_results_complement = eval_model(
                circuit_gpt, eval_dl, tokenizer, device, use_weight_mask=True, use_edge_mask=False, reverse=True
            )
            print(
                "Weight pruning epoch {}, discovered circuit accuracy: {:.4f}, complementary circuit accuracy: {:.4f}, KL: {:.4f}, weight density: {:.4f}, edge density: {:.4f}".format(
                    epoch + 1, eval_results_faith['acc'], eval_results_complement['acc'], eval_results_faith['kl'], eval_results_faith['weight_density'], eval_results_faith['edge_density'])
            )
            # save good weight masks
            if eval_results_faith['acc'] > 0.95 and eval_results_faith['weight_density'] < 0.05:
                print(f'saving weight masks at epoch {epoch}')
                mask_logits_dict_weight = {k:v.detach().cpu() for k,v in circuit_gpt.mask_logits_dict_weight.items()}
                torch.save(
                    circuit_gpt.mask_logits_dict_weight,
                    join(args.results_dir, f'mask_logits_dict_weight_ioi_{args.ds_idx}_weight_{epoch}_edge_0.pt')
                )
    else:
        n_epoch_w = args.resume_epoch_w

    
    if args.prune_edge:
        edge_logits = [mask for _, mask in circuit_gpt.mask_logits_dict_edge.items()]    
        optim_edge = torch.optim.AdamW(edge_logits, lr=args.lr_edge)
        if args.use_weight_masks:
            circuit_gpt.turn_on_weight_masks(deterministic=True)
        
        for epoch in tqdm(range(args.train_epochs_edge)):
            lambda_sparse_edge = schedule_epoch_lambda(
                epoch, 
                lambda_0=args.lambda_sparse_edge_init,
                max_times=args.max_times_lambda_sparse_edge, 
                min_times=args.min_times_lambda_sparse_edge, 
                n_epoch_warmup=args.n_epoch_warmup_lambda_sparse_edge,
                n_epoch_cooldown=args.n_epoch_cooldown_lambda_sparse_edge,
            )
            # lambda_complete_edge = schedule_epoch_lambda(epoch, args.lambda_complete_edge_init)
            lambda_complete_edge = args.lambda_complete_edge_init
            # if epoch > 1:
            #     if eval_results_faith['edge_density'] <= 0.05:
            #         for g in optim_edge.param_groups:
            #             g['lr'] = 0.01
            
            for batch in train_dl:
                batch_inputs = prepare_batch_inputs(batch, tokenizer)
                
                circuit_gpt.turn_on_edge_masks(deterministic=False)
                sparse_loss_edge = circuit_gpt.edge_sparseness_loss()
            
                batch_logits_masked = circuit_gpt(batch_inputs['input_ids'].to(device))[0] 
                faith_loss_edge, _, _ = compute_faith_loss(batch_logits_masked, batch_inputs)

                if lambda_complete_edge > 0:
                    circuit_gpt.turn_on_edge_masks(deterministic=False, reverse=True)
                    batch_logits = circuit_gpt(batch_inputs['input_ids'].to(device))[0] 
                    complete_loss_edge, _ = compute_complete_loss(batch_logits, batch_inputs) 
                else:
                    complete_loss_edge = 0.
                
                loss_edge = faith_loss_edge + sparse_loss_edge *  lambda_sparse_edge + complete_loss_edge * lambda_complete_edge
                loss_edge.backward()
                optim_edge.step()
                optim_edge.zero_grad()
                torch.cuda.empty_cache()
        
            eval_results_faith = eval_model(
                circuit_gpt, eval_dl, tokenizer, device, use_weight_mask=True, use_edge_mask=True, reverse=False
            )
            eval_results_complement = eval_model(
                circuit_gpt, eval_dl, tokenizer, device, use_weight_mask=True, use_edge_mask=True, reverse=True
            )
            print(
                "Edge pruning epoch {}, discovered circuit accuracy: {:.4f}, complementary circuit accuracy: {:.4f}, KL: {:.4f}, weight density: {:.4f}, edge density: {:.4f}".format(
                    epoch + 1, eval_results_faith['acc'], eval_results_complement['acc'], eval_results_faith['kl'], eval_results_faith['weight_density'], eval_results_faith['edge_density'])
            )
        
            # save good edge masks
            if eval_results_faith['acc'] > 0.95 and eval_results_faith['edge_density'] < 0.05:
                print(f'saving edge masks at epoch {epoch}')
                mask_logits_dict_edge = {k:v.detach().cpu() for k,v in circuit_gpt.mask_logits_dict_edge.items()}
                torch.save(
                    circuit_gpt.mask_logits_dict_edge,
                    join(args.results_dir, f'mask_logits_dict_edge_ioi_{args.ds_idx}_weight_{n_epoch_w}_edge_{epoch}.pt')
                )
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='/home/leiyu/scratch/circuit-discovery/mask_logits/', type=str)
    parser.add_argument('--data_dir', default='/home/leiyu/projects/def-yangxu/leiyu/circuit-discovery/data/', type=str)
    parser.add_argument('--model_dir', default='/home/leiyu/projects/def-yangxu/leiyu/LMs/', type=str)
    parser.add_argument('--model_name', default='gpt2-small', type=str)
    parser.add_argument('--prune_weight', action='store_true')
    parser.add_argument('--use_weight_masks', action='store_true')
    parser.add_argument('--prune_edge', action='store_true')
    
    parser.add_argument('--gs_temp_weight', default=0.01, type=float)
    parser.add_argument('--gs_temp_edge', default=1.0, type=float)
    parser.add_argument('--logits_w_init', default=0.0, type=float)
    parser.add_argument('--logits_e_init', default=0.0, type=float)
    parser.add_argument('--lr_weight', default=0.1, type=float)
    parser.add_argument('--lr_edge', default=0.1, type=float)
    parser.add_argument('--lambda_sparse_weight_init', default=1.0, type=float)
    parser.add_argument('--lambda_sparse_edge_init', default=1.0, type=float)
    parser.add_argument('--lambda_complete_weight_init', default=1.0, type=float)
    parser.add_argument('--lambda_complete_edge_init', default=1.0, type=float)
    parser.add_argument('--max_times_lambda_sparse_weight', default=1000., type=float)
    parser.add_argument('--min_times_lambda_sparse_weight', default=1., type=float)
    parser.add_argument('--max_times_lambda_sparse_edge', default=100., type=float)
    parser.add_argument('--min_times_lambda_sparse_edge', default=0.01, type=float)

    parser.add_argument('--n_ioi_data', default=1280, type=int)
    parser.add_argument('--ds_idx', default=0, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_epochs_weight', default=300, type=int)
    parser.add_argument('--train_epochs_edge', default=100, type=int)
    parser.add_argument('--n_epoch_warmup_lambda_sparse_weight', default=500, type=int)
    parser.add_argument('--n_epoch_cooldown_lambda_sparse_weight', default=1, type=int)
    parser.add_argument('--n_epoch_warmup_lambda_sparse_edge', default=20, type=int)
    parser.add_argument('--n_epoch_cooldown_lambda_sparse_edge', default=20, type=int)
    parser.add_argument('--resume_epoch_w', default=0, type=int)
    parser.add_argument('--resume_epoch_e', default=0, type=int)
    

    args = parser.parse_args()
    main(args)
