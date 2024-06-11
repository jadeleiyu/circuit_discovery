from dataclasses import dataclass

import torch

from transformer_lens import HookedTransformerConfig
import transformer_lens.loading_from_pretrained as loading

def get_config(model_name, **kwargs):
    official_model_name = loading.get_official_model_name(model_name)
    tl_config = loading.get_pretrained_model_config(
        official_model_name,
        hf_cfg=None,
        checkpoint_index=None,
        checkpoint_value=None,
        fold_ln=True,
        # device=device,
        # n_devices=n_devices,
        default_prepend_bos=True,
        dtype=torch.float32,
        # **from_pretrained_kwargs,
    )

    tl_config_dict = tl_config.to_dict()
    tl_config_dict.update(kwargs)
    return CircuitLMConfig.from_dict(tl_config_dict)

@dataclass
class CircuitLMConfig(HookedTransformerConfig):
    debug: bool = False
    gs_temp_weight: float = 1.
    gs_temp_edge: float = 1.
    logits_w_init: float = 1.
    logits_e_init: float = 1.
    use_weight_masks: bool = True # false means not registering mask logits for gpt weights
    use_edge_masks: bool = True 
    use_deterministic_masks: bool = False
    use_reverse_masks: bool = False
