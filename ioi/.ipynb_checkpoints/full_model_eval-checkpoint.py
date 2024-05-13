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


def main(args):

    # path that stores gpt-small weights and gpt tokenizer
    model_path = join(args.model_dir, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # load IOI data
    # Note that there are overlaps between train and test sets, 
    # due to the way IOIDataset is constructed (randomly sample N items)
    ds = IOIGeneratorDataset(prompt_type="ABBA", N=args.n_ioi_data, tokenizer=tokenizer)
    ioi_ds = IOIFullModelDataset(prepare_ioi_data_for_clm(ds.ioi_prompts))
    ioi_dl = DataLoader(ioi_ds, batch_size=args.batch_size)

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

    for batch in ioi_dl:
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
    full_model_acc = 1. - (full_model_pred_labels.sum() / len(full_model_pred_labels)).item()

    print(f'ioi dataset {args.ds_idx}')
    print(f'full_model_target_log_probs: {full_model_target_log_probs.shape}')
    print(f'full_model_pred_labels: {full_model_pred_labels.shape}')
    print(f'full model accuracy: {full_model_acc}')
        
    torch.save(full_model_target_log_probs, f'full_model_results/target_log_probs_{args.ds_idx}.pt')
    torch.save(full_model_pred_labels, f'full_model_results/pred_labels_{args.ds_idx}.pt')
    pickle.dump(ds.ioi_prompts, open(join(args.data_dir, f'ioi_prompts_{args.ds_idx}.p'), 'wb'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/leiyu/projects/def-yangxu/leiyu/circuit-discovery/data/', type=str)
    parser.add_argument('--ds_idx', default=0, type=int)
    parser.add_argument('--model_dir', default='/home/leiyu/projects/def-yangxu/leiyu/LMs/', type=str)
    parser.add_argument('--model_name', default='gpt2-small', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_ioi_data', default=640, type=int)
    
    parser.add_argument('--gs_temp_weight', default=0.01, type=float)
    parser.add_argument('--gs_temp_edge', default=1.0, type=float)
    parser.add_argument('--logits_w_init', default=0.0, type=float)
    parser.add_argument('--logits_e_init', default=0.0, type=float)
    

    args = parser.parse_args()
    main(args)
    






















