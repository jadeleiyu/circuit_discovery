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

class FullModelAgreementDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['prompt'])

    def __getitem__(self, i):

        return {
            'prompt': self.data['prompt'][i],
            'target good': ' ' + self.data['targets'][i][0],
            'target bad': ' ' + self.data['targets'][i][1]
        }


def prepare_data_for_clm(ds):
    sents = []
    targets = []

    for row in ds:
        sg, sb = row['sentence_good'][:-1].split(), row['sentence_bad'][:-1].split()
        combined = []
        target_good, target_bad = None, None
        has_got_full_prefix = False
        for i, (tg, tb) in enumerate(zip(sg, sb)):

            if tg == tb:
                combined.append(tg)
                # print(combined)
            else:
                has_got_full_prefix = True
                # combined.append('[MASK]')
                target_good, target_bad = tg, tb

            if not has_got_full_prefix:
                continue

        sent = ' '.join(combined)
        sents.append(sent)
        targets.append((target_good, target_bad))

    data_dict = {}
    data_dict['prompt'] = sents

    # data_dict['mask_token_ids'] = [x.index(tokenizer.mask_token_id) for x in data_dict['input_ids']]
    data_dict['targets'] = targets

    return data_dict


def prepare_batch_inputs(batch, tokenizer):
    batch_tokenized = tokenizer(
        batch['prompt'],
        return_tensors="pt",
        padding=True
    )
    batch_seq_len = batch_tokenized['attention_mask'].sum(-1)
    batch_target_good = torch.tensor([
        token_ids[0] for token_ids in tokenizer(batch['target good'])['input_ids']
    ])
    batch_target_bad = torch.tensor([
        token_ids[0] for token_ids in tokenizer(batch['target bad'])['input_ids']
    ])

    return {
        'input_ids': batch_tokenized['input_ids'],
        'seq_lens': batch_seq_len,
        'target good': batch_target_good,
        'target bad': batch_target_bad
    }


def main(args):

    # path that stores gpt-small weights and gpt tokenizer
    model_path = join(args.model_dir, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # load agreement data
    ds_path = args.data_dir + args.task_name
    raw_agreement_ds = load_from_disk(ds_path)['train']
    processed_agreement_ds = prepare_data_for_clm(raw_agreement_ds)
    agreement_ds = FullModelAgreementDataset(processed_agreement_ds)
    agreement_dl = DataLoader(
        agreement_ds,
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
        use_weight_masks=False,
        use_edge_masks=False
    )
    circuit_gpt = CircuitGPT(circuit_gpt_config)
    circuit_gpt.load_pretrained_weight(gpt_weights)
    circuit_gpt.to(device);

    circuit_gpt.eval()

    full_model_target_log_probs = []
    full_model_pred_labels = []
    # for _, batch in tqdm(enumerate(agreement_dl)):
    for batch in agreement_dl:
        batch_inputs = prepare_batch_inputs(batch, tokenizer)
        batch_seq_lens = batch_inputs['seq_lens']
        batch_size = batch_inputs['input_ids'].shape[0]
        with torch.no_grad():
            batch_logits = circuit_gpt(batch_inputs['input_ids'].to(device))[0]  # (B, seq_len, vocab_size)
        logits_target_good = batch_logits[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target good']]
        logits_target_bad = batch_logits[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target bad']]
        logits_gb = torch.stack([logits_target_good, logits_target_bad], -1)  # (B,2)
        
        full_model_target_log_probs.append(
            F.log_softmax(logits_gb, -1).cpu()
        )   # (B, vocab_size)
        full_model_pred_labels.append(
            (logits_target_good < logits_target_bad).long().cpu()
        )

    full_model_target_log_probs = torch.cat(full_model_target_log_probs, 0) # (n_data, 2)
    full_model_pred_labels = torch.cat(full_model_pred_labels, 0) # (n_data, )
    print(f'full_model_target_log_probs: {full_model_target_log_probs.shape}')
    print(f'full_model_pred_labels: {full_model_pred_labels.shape}')
        
    torch.save(full_model_target_log_probs, f'full_model_results/{args.task_name}_target_log_probs.pt')
    torch.save(full_model_pred_labels, f'full_model_results/{args.task_name}_pred_labels.pt')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/leiyu/projects/def-yangxu/leiyu/circuit-discovery/data/', type=str)
    parser.add_argument('--task_name', default='anaphor_gender_agreement', type=str)
    parser.add_argument('--model_dir', default='/home/leiyu/projects/def-yangxu/leiyu/LMs/', type=str)
    parser.add_argument('--model_name', default='gpt2-small', type=str)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--gs_temp_weight', default=0.01, type=float)
    parser.add_argument('--gs_temp_edge', default=1.0, type=float)
    parser.add_argument('--logits_w_init', default=0.0, type=float)
    parser.add_argument('--logits_e_init', default=0.0, type=float)
    

    args = parser.parse_args()
    main(args)
    






















