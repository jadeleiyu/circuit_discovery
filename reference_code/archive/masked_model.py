import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertLMHeadModel


def gumbel_sigmoid(logits, gs_temp=1., eps=1e-10):
    uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0, 1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / gs_temp)
    res = ((res > 0.5).type_as(res) - res).detach() + res
    return res


class BERTProbDiff(BertLMHeadModel):
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
        mask_token_ids=None, targets=None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        bs = torch.arange(len(input_ids))

        mask_token_logits = self.cls(outputs[0][bs, mask_token_ids, :])

        return torch.stack([
            mask_token_logits[bs, targets[:, 0]], mask_token_logits[bs, targets[:, 1]]
        ], -1)  # (B, 2)

        # return mask_token_logits
        # softmaxed = mask_token_logits.softmax(1)
        # return softmaxed[bs, targets[:, 0]] - softmaxed[bs, targets[:, 1]]


class MaskedModel(nn.Module):
    def __init__(self, model, args, mask_logits_dict=None):
        super().__init__()
        self.model = model
        self.mask_param_names = args['param_keys']
        self.mask_logits = {}
        self.masks = {}
        self.unmasked_params = {}  # a copy of the unmasked params
        self.is_masked = False  # no masks applied upon initialization
        self.gs_temp = args['temperature_weight']

        self.param_names = [n for n, _ in list(
            model.named_parameters())]  # full list of param names of the transformer (some may not be masked)
        self.model_param_list = [p for _, p in list(
            model.named_parameters())]  # full list of params of the transformer (some may not be masked)
        self.param_name2param_id = {n: i for i, n in enumerate(self.param_names)}

        for n, p in self.model.named_parameters():
            # todo: do not disable parameters for edge masks 
            if 'edge' not in n:
                p.grad = None
                p.requires_grad = False
            
        for param_name in self.mask_param_names:
            param_id = self.param_name2param_id[param_name]
            m = self.model_param_list[self.param_name2param_id[param_name]]
            # trainable parameters are the logits of the binary-concrete distribution
            if mask_logits_dict is None:
                masks_logits = nn.Parameter(
                    torch.nn.init.normal_(torch.ones_like(m), mean=2.7, std=0.01),
                    requires_grad=True)
            else:
                masks_logits = nn.Parameter(mask_logits_dict[param_id], requires_grad=True)
            self.mask_logits[param_id] = masks_logits
            self.masks[param_id] = torch.ones_like(m.detach())
            # save a copy of the unmasked params so that we can recover them after each masked forward run
            self.unmasked_params[param_id] = m.detach().clone()

    def apply_masks(self, reverse_mask=False, deterministic_masks=False):

        for n in self.mask_param_names:
            param_id = self.param_name2param_id[n]
            m = self.model_param_list[param_id]
            unmasked_m = self.unmasked_params[param_id]
            if not deterministic_masks:
                sampled_masks = gumbel_sigmoid(self.mask_logits[param_id], gs_temp=self.gs_temp)
            else:
                with torch.no_grad():
                    sampled_masks = torch.where(self.mask_logits[param_id] > 0., 1., 0.)
            if reverse_mask:
                sampled_masks = 1 - sampled_masks
            self.masks[param_id] = sampled_masks.detach().clone()
            m.copy_(sampled_masks * unmasked_m)
            
        self.is_masked = True

    def remove_masks(self):
        for n in self.mask_param_names:
            param_id = self.param_name2param_id[n]
            m = self.model_param_list[param_id]
            unmasked_m = self.unmasked_params[param_id]
            m.copy_(unmasked_m)
            m.detach_()
            self.masks[param_id] = torch.ones_like(unmasked_m)
        self.is_masked = False

    def turn_on_mask_grads(self):
        for _, mask_logits in  self.mask_logits:
            mask_logits.requires_grad = True
            
    def turn_off_mask_grads(self):
        for _, mask_logits in  self.mask_logits:
            mask_logits.requires_grad = False

    def forward(self, input_ids):
        return self.model.forward(input_ids)

    def get_trainable_parameters(self):
        return (mask_logits for _, mask_logits in self.mask_logits.items())

    def get_sparseness_loss(self):
        sparse_losses = []
        n_param = 0
        for _, mask_logits in self.mask_logits.items():
            sparse_loss_k = F.sigmoid(mask_logits).sum()
            sparse_losses.append(sparse_loss_k)
            n_param += torch.ones_like(mask_logits.detach().clone()).sum()
        return torch.stack(sparse_losses).sum() / n_param

    
    def get_pruned_model_density(self):
        with torch.no_grad():
            n_param, n_open_mask = 0, 0
            for _, mask in self.masks.items():
                n_param += torch.ones_like(mask).sum()
                n_open_mask += mask.sum()
            weight_den = n_open_mask / n_param
            return n_param.item(), n_open_mask.item(), weight_den.item()
            

    def save_mask_logits(self, fn):
        mask_logits = {k: v.detach().cpu() for k, v in self.mask_logits.items()}
        torch.save(mask_logits, fn)


    def load_mask_logits(self, fn, device):

        mask_logits = torch.load(fn)
        for k in mask_logits.keys():
            mask_logits[k] = mask_logits[k].to(device)
            mask_logits[k].requires_grad=True
        self.mask_logits = mask_logits



        
