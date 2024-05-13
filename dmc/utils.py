import torch
import torch.nn as nn


def get_batch_logits_clm(output_logits, batch_inputs):
    # batch_logits: shape (B, vocab_size)
    bs = torch.arange(output_logits.shape[0])
    return torch.stack([
        output_logits[bs, batch_inputs['target good']], output_logits[bs, batch_inputs['target bad']]
    ], -1)  # (B, 2)


    
def compute_faith_loss_clm(batch_logits, batch_inputs):
    # batch_logits: (B, seq_len, vocab_size)
    batch_seq_lens = batch_inputs['seq_lens']
    batch_size = batch_logits.shape[0]
    
    logits_target_good = batch_logits[torch.arange(batch_size), batch_seq_lens-1, batch_inputs['target good']]
    logits_target_bad = batch_logits[torch.arange(batch_size), batch_seq_lens-1, batch_inputs['target bad']]
    logits_gb = torch.stack([logits_target_good, logits_target_bad], -1) # (B,2)

    batch_unmasked_pred_labels = batch_inputs['unmasked_pred_label'].to(logits_gb.device)
    batch_faith_loss = nn.functional.cross_entropy(logits_gb, batch_unmasked_pred_labels)

    return batch_faith_loss, logits_gb



def compute_faith_loss_accept(batch_logits_good, batch_logits_bad, batch_inputs):
    # batch_logits: (B, seq_len, vocab_size)
    batch_log_probs_good = get_seq_log_probs(batch_logits_good, batch_inputs['input_ids_good'], batch_inputs['seq_lens_good'])
    batch_log_probs_bad = get_seq_log_probs(batch_logits_bad, batch_inputs['input_ids_bad'], batch_inputs['seq_lens_bad'])

    batch_log_probs = torch.stack([batch_log_probs_good, batch_log_probs_bad], -1) # (B,2)
    batch_unmasked_pred_labels = batch_inputs['unmasked_pred_label'].to(batch_log_probs.device)
    batch_faith_loss = nn.functional.cross_entropy(batch_log_probs, batch_unmasked_pred_labels)

    return batch_faith_loss, batch_log_probs


def get_seq_log_probs(batch_logits, batch_inputs, batch_seq_lens):
    # batch_logits: (B, seq_len, vocab_size)
    # batch_inputs: (B, seq_len)
    batch_inputs_ids = batch_inputs[:,1:].unsqueeze(-1).to(batch_logits.device)  # (B, seq_len-1)
    batch_log_probs = nn.functional.log_softmax(batch_logits, -1)[:,:-1]  # (B, seq_len-1, vocab_size)
    batch_token_log_probs = torch.gather(
        batch_log_probs, -1, batch_inputs_ids
    ).squeeze()  # (B, seq_len-1)

    batch_seq_log_probs = []
    for i in range(batch_token_log_probs.shape[0]):
        batch_seq_log_probs.append(
            batch_token_log_probs[i][:batch_seq_lens[i]-1].sum()
        )
    
    return torch.stack(batch_seq_log_probs)  # (B, )



def get_target_module_keys(model):
    keys = []
    for n, p in model.named_parameters():
        if 'blocks.' in n and 'edge' not in n:
            if len(p.shape) == 2:
                keys.append(n)
            elif 'bias' in n:
                keys.append(n)
    return keys


def get_lambda_sparse_edge(epoch):
    if epoch < 20:
        return 0.01 * epoch / 20
    elif epoch < 100:
        return 0.01 - 0.009 * epoch / 100
    else:
        return 0.001


def get_lambda_sparse_weight(epoch, args):
    if epoch <= 40:
        return args['lambda_sparse_weight']
    else:
        return args['lambda_sparse_weight'] + (epoch - 40)*0.1










