from torch.utils.data import Dataset


class OQACircuitDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['prompt'])
        
    def __getitem__(self, i):
        return {
            'prompt': self.data['prompt'][i].strip(),
            'label': self.data['label'][i],
            'full_model_target_log_probs': self.data['full_model_target_log_probs'][i],
            'full_model_pred_label': self.data['full_model_pred_labels'][i],
        }


def prepare_batch_inputs(batch, tokenizer):
    batch_inputs = tokenizer(
        batch['prompt'], return_tensors='pt', padding=True
    )
    
    batch_seq_lens = batch_inputs.attention_mask.sum(-1)

    if 'full_model_target_log_probs' not in batch:
        return {
            'input_ids': batch_inputs.input_ids,
            'seq_lens': batch_seq_lens,
            'label': batch['label']
        }
        
    else:
        return {
            'input_ids': batch_inputs.input_ids,
            'seq_lens': batch_seq_lens,
            'label': batch['label'],
            'full_model_target_log_probs': batch['full_model_target_log_probs'],
            'full_model_pred_label': batch['full_model_pred_label'],
        }
        

