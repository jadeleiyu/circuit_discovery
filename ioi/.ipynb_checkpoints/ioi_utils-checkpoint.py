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

from ioi_dataset import *


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
    # batch_logits: (B, seq_len, vocab_size)
    batch_seq_lens = batch_inputs['seq_lens']
    batch_size = batch_logits_masked.shape[0]
    log_probs_target_unmasked = batch_inputs['full_model_target_log_probs']  # (B, 2
    # print(f'batch_logits_masked: {batch_logits_masked.shape}')
    # print(f'log_probs_target_unmasked: {log_probs_target_unmasked.shape}')
    # print()
    # log_probs_target_unmasked = F.log_softmax(logits_gb_unmasked, -1)  # (B, 2)

    logits_target_good_masked = batch_logits_masked[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target good']]
    logits_target_bad_masked = batch_logits_masked[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target bad']]
    logits_gb_masked = torch.stack([logits_target_good_masked, logits_target_bad_masked], -1)  # (B,2)
    # log_probs_target_masked = F.log_softmax(batch_logits_masked[:, batch_seq_lens - 1], -1)  # (B, vocab_size)
    log_probs_target_masked = F.log_softmax(logits_gb_masked, -1)  # (B, 2)

    # batch_labels = torch.zeros(batch_size).long().to(logits_gb.device)
    batch_labels = batch_inputs['full_model_pred_label'].to(logits_gb_masked.device)
    batch_pred = (logits_gb_masked[:, 0] < logits_gb_masked[:, 1]).long()
    batch_faith_loss = F.cross_entropy(logits_gb_masked, batch_labels)
    batch_kl = F.kl_div(
        log_probs_target_unmasked.to(log_probs_target_masked.device), 
        log_probs_target_masked,
        log_target=True
    ).cpu()
    batch_n_correct = (batch_labels == batch_pred).sum().cpu().item()

    return batch_faith_loss, batch_kl, batch_n_correct


def compute_complete_loss(batch_logits, batch_inputs):
    # batch_logits: (B, seq_len, vocab_size)
    batch_seq_lens = batch_inputs['seq_lens']
    batch_size = batch_logits.shape[0]

    logits_target_good = batch_logits[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target good']]
    logits_target_bad = batch_logits[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target bad']]
    logits_gb = torch.stack([logits_target_good, logits_target_bad], -1)  # (B,2)

    batch_probs_uniform = torch.ones(logits_gb.shape).to(logits_gb.device) * 0.5
    batch_complete_loss = nn.functional.cross_entropy(logits_gb, batch_probs_uniform)

    return batch_complete_loss, logits_gb


@torch.no_grad()
def eval_model(model, eval_dl, tokenizer, device, 
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
        batch_logits_masked = model(batch_inputs['input_ids'].to(device))[0]  # (B, seq_len, vocab_size)
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
        'edge_density': edge_density,
        'n_correct': correct,
        'total': total
    }

