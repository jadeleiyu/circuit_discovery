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

from oqa_dataset import *


# sparsity loss weight scheduling function
# now using a linear scheduler by first increasing lambda_sparse from lambda_0 to max_times * lambda_0 in n_epoch_warmup epochs,
# and then reducing lambda_sparse to min_times * lambda_0 in another n_epoch_cooldown epochs
# the current ROT is to first heavily penalize edge density through high lr and lambda_sparse
# and then decrease lambda_sparse to recover essential edges for IOI
# therefore evaluation acc will first drop to near/below random and then go back to near-perfect
def schedule_epoch_lambda(epoch, lambda_0, max_times=100., min_times=0.001,
                          n_epoch_warmup=20, n_epoch_cooldown=20):

    if epoch < n_epoch_warmup:
        return lambda_0  + lambda_0 * (max_times - 1) * epoch / n_epoch_warmup
        
    elif epoch < n_epoch_warmup + n_epoch_cooldown:
        return lambda_0 * max_times - lambda_0 * (max_times - min_times) * (epoch - n_epoch_warmup) / n_epoch_cooldown
        
    else:
        return lambda_0 * min_times


def compute_faith_loss(batch_logits_masked, batch_inputs):
    # batch_logits: (B, seq_len, cap_vocab_size)
    batch_seq_lens = batch_inputs['seq_lens']
    batch_size = batch_logits_masked.shape[0]
    log_probs_target_unmasked = batch_inputs['full_model_target_log_probs'].to(batch_logits_masked.device)  # (B, cap_vocab_size)
    # batch_labels = batch_inputs['full_model_pred_label'].to(batch_logits_masked.device) # (B)
    batch_labels = batch_inputs['label'].to(batch_logits_masked.device)

    batch_logits_masked_target = batch_logits_masked[torch.arange(batch_size), batch_seq_lens - 1]  # (B, cap_vocab_size)
    
    batch_faith_loss = F.cross_entropy(batch_logits_masked_target, batch_labels)
    with torch.no_grad():
        batch_kl = F.kl_div(
            log_probs_target_unmasked.to(batch_logits_masked.device), 
            F.log_softmax(batch_logits_masked_target, -1),
            log_target=True
        ).cpu()
        batch_pred = torch.argsort(batch_logits_masked_target, -1)[:, -1]  # (B)
        batch_n_correct = (batch_labels == batch_pred).sum().cpu().item()

    return batch_faith_loss, batch_kl, batch_n_correct


def compute_complete_loss(batch_logits_masked, batch_inputs):
    # batch_logits: (B, seq_len, cap_vocab_size)
    batch_seq_lens = batch_inputs['seq_lens']
    batch_size = batch_logits_masked.shape[0]

    batch_logits_masked_target = batch_logits_masked[torch.arange(batch_size), batch_seq_lens - 1]  # (B, cap_vocab_size)

    batch_probs_uniform = torch.ones(batch_logits_masked_target.shape).to(batch_logits_masked_target.device) * (1. / batch_logits_masked_target.shape[-1])
    batch_complete_loss = nn.functional.cross_entropy(batch_logits_masked_target, batch_probs_uniform)

    return batch_complete_loss, batch_logits_masked_target


@torch.no_grad()
def eval_model(model, eval_dl, tokenizer, device, capital_vocab_idx,
               use_weight_mask=True, use_edge_mask=True, reverse=False):
    
    model.eval()

    # get weight and edge density     
    model.turn_on_weight_masks(deterministic=True, reverse=reverse)  
    _, _, weight_density = model.get_weight_density()       
    model.turn_on_edge_masks(deterministic=True, reverse=reverse)  
    _, _, edge_density = model.get_edge_density()

    if not use_weight_mask:
        model.turn_off_weight_masks()
    if not use_edge_mask:     
        model.turn_off_edge_masks()

    total = len(eval_dl.dataset)
    correct = 0
    kls = []
    faith_losses = []

    for batch in eval_dl:
        batch_inputs = prepare_batch_inputs(batch, tokenizer)       
        batch_logits_masked = model(batch_inputs['input_ids'].to(device))[0][:,:,capital_vocab_idx]  # (B, seq_len, cap_vocab_size)
        batch_faith_loss, batch_kl, batch_n_correct = compute_faith_loss(batch_logits_masked, batch_inputs)
        
        # print(batch_logits_gb)
        correct += batch_n_correct
        kls.append(batch_kl)
        faith_losses.append(batch_faith_loss.cpu())

        torch.cuda.empty_cache()

    model.turn_off_weight_masks()
    model.turn_off_edge_masks()

    acc = correct / total

    return {
        'acc': acc,
        'kl': torch.stack(kls).mean().item(),
        'faith_loss': torch.stack(faith_losses).mean().item(),
        'weight_density': weight_density,
        'edge_density': edge_density
    }
