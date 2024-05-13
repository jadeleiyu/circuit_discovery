from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import pickle
import transformer_lens
from torch.optim import AdamW
from os.path import join
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import argparse
import json

import sys
sys.path.append('/home/leiyu/projects/def-yangxu/leiyu/circuit-discovery')
from dmc.circuit_gpt import *
from oqa_dataset import *
from tqdm import tqdm   # why must import this at the end? otherwise there's a module type error
from datasets import Dataset


def build_ds(pararel_rel_data, rel_id):
    data = pararel_rel_data[rel_id]
    answer_vocab = []
    ds_dict = {
        'prompt': [],
        'answer': [],
    }
    
    for entry in data:
        prompt = entry[0][0].replace(' [MASK] .', '')
        prompt = prompt.replace(' [MASK].', '')
        if '[MASK]' not in prompt:
            target = entry[0][1]
            ds_dict['prompt'].append(prompt)
            ds_dict['answer'].append(' ' + target)
            answer_vocab.append(' ' + target)
    
    answer_vocab = list(set(ds_dict['answer']))
    answer_vocab_idx = torch.tensor([
        input_ids[0] for input_ids in tokenizer(answer_vocab).input_ids
    ])
    
    answer_vocab2class_id = {
        answer_vocab_idx[i].item():i for i in range(len(answer_vocab_idx)) 
    }
    
    answer2vocab_id = {answer: answer_vocab_id.item() for answer, answer_vocab_id in zip(answer_vocab, answer_vocab_idx)}
    
    ds_dict['label'] = [
        answer_vocab2class_id[answer2vocab_id[answer]] for answer in ds_dict['answer']
    ]   
    
    ds = Dataset.from_dict(ds_dict)
    # dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    return ds

def main(args):

    # path that stores gpt-small weights and gpt tokenizer
    model_path = join(args.model_dir, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    ########################
    # load OQA data
    with open(join(args.data_dir, 'pararel_data_all.json')) as open_file:
        pararel_rel_data = json.load(open_file)      
    rel_ids = args.pararel_rel_ids.split(', ')
    rel_ids = [rel_id for rel_id in rel_ids if rel_id in pararel_rel_data]

    for rel in rel_ids:
        
    
    data = []
    for rel_id in rel_ids:
        try:
            data += pararel_rel_data[rel_id]
        except KeyError as e:
            pass
    
    answer_vocab = []
    ds_dict = {
        'prompt': [],
        'answer': [],
    }
    
    for entry in data:
        prompt = entry[0][0].replace(' [MASK] .', '')
        prompt = prompt.replace(' [MASK].', '')
        if '[MASK]' not in prompt:
            target = entry[0][1]
            ds_dict['prompt'].append(prompt)
            ds_dict['answer'].append(' ' + target)
            answer_vocab.append(' ' + target)
    
    answer_vocab = list(set(ds_dict['answer']))
    answer_vocab_idx = torch.tensor([
        input_ids[0] for input_ids in tokenizer(answer_vocab).input_ids
    ])
    
    answer_vocab2class_id = {
        answer_vocab_idx[i].item():i for i in range(len(answer_vocab_idx)) 
    }
    
    answer2vocab_id = {answer: answer_vocab_id.item() for answer, answer_vocab_id in zip(answer_vocab, answer_vocab_idx)}
    
    ds_dict['label'] = [
        answer_vocab2class_id[answer2vocab_id[answer]] for answer in ds_dict['answer']
    ]   
    
    ds = Dataset.from_dict(ds_dict)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    # print(next(iter(dl)))
    ##########################

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

    n_correct = 0
    n_batch = int(len(dl.dataset) / dl.batch_size) + 1

    for batch in tqdm(dl, total=n_batch):
        batch_inputs = prepare_batch_inputs(batch, tokenizer)
        batch_seq_lens = batch_inputs['seq_lens']
        batch_size = batch_inputs['input_ids'].shape[0]
        with torch.no_grad():
            batch_logits = circuit_gpt(batch_inputs['input_ids'].to(device))[0]  # (B, seq_len, vocab_size)
        batch_logits_next_tok = batch_logits[torch.arange(batch_size), batch_seq_lens - 1][:, answer_vocab_idx]  # (B, n_capital_class)
        batch_pred_cap_ids = torch.argsort(batch_logits_next_tok, -1)[:, -1].cpu()  # (B)
        
        full_model_target_log_probs.append(
            F.log_softmax(batch_logits_next_tok, -1).cpu()
        )   # (B, n_capital_class)
        full_model_pred_labels.append(batch_pred_cap_ids)  # (B)
        n_correct += (batch_pred_cap_ids == batch['label']).sum().item()

    full_model_target_log_probs = torch.cat(full_model_target_log_probs, 0) # (n_data, n_capital_class)
    full_model_pred_labels = torch.cat(full_model_pred_labels, 0) # (n_data, )
    full_model_acc = float(n_correct) / len(ds)

    print(f'pararel relations: {rel_ids}')
    print(f'number of questions: {len(ds)}')
    print(f'full_model_target_log_probs: {full_model_target_log_probs.shape}')
    print(f'full_model_pred_labels: {full_model_pred_labels.shape}')
    print(f'full model accuracy: {full_model_acc}')
        
    torch.save(full_model_target_log_probs, f'full_model_results/target_log_probs.pt')
    torch.save(full_model_pred_labels, f'full_model_results/pred_labels.pt')
    torch.save(answer_vocab_idx, f'full_model_results/answer_vocab_idx.pt')
    pickle.dump(ds_dict, open(join(args.data_dir, 'pararel_ds_dict.p'), 'wb'))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/leiyu/projects/def-yangxu/leiyu/circuit-discovery/data/', type=str)
    parser.add_argument('--model_dir', default='/home/leiyu/projects/def-yangxu/leiyu/LMs/', type=str)
    parser.add_argument('--model_name', default='gpt2-small', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--pararel_rel_ids', default='P103, P127, P131, P136, P138, P140, P159, P17, P176, P19, P20, P264, P276, P279, P30, P361, P364, P37, P407, P413, P449, P495, P740, P1376, P36', type=str)
    
    parser.add_argument('--gs_temp_weight', default=0.01, type=float)
    parser.add_argument('--gs_temp_edge', default=1.0, type=float)
    parser.add_argument('--logits_w_init', default=0.0, type=float)
    parser.add_argument('--logits_e_init', default=0.0, type=float)
    

    args = parser.parse_args()
    main(args)
    






















