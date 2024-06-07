from torch.utils.data import Dataset
import torch


class AgreementDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['prompt'])

    def __getitem__(self, i):

        return {
            'prompt': self.data['prompt'][i],
            'target good': ' ' + self.data['targets'][i][0],
            'target bad': ' ' + self.data['targets'][i][1],
            'full_model_target_log_probs': self.data['full_model_target_log_probs'][i],
            'full_model_pred_label': self.data['full_model_pred_label'][i]
        }

def prepare_data_for_clm(ds, full_model_target_log_probs, full_model_pred_labels):
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

    data_dict['full_model_target_log_probs'] = full_model_target_log_probs
    data_dict['full_model_pred_label'] = full_model_pred_labels

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
        'target bad': batch_target_bad,
        'full_model_target_log_probs': batch['full_model_target_log_probs'],
        'full_model_pred_label': batch['full_model_pred_label'],
    }

