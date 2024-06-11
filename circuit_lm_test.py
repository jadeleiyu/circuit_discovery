from pathlib import Path

from disco_gp.circuit_lm import CircuitTransformer
from disco_gp.circuit_lm_config import get_config

from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

import torch

if __name__ == '__main__':

    model_name = 'meta-llama/Llama-2-7b-hf'
    # model_name = 'gpt2'


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    t = tokenizer('Colorless green ideas sleep furiously', return_tensors='pt')
    tt = t.input_ids.to('cuda')

    model = CircuitTransformer.from_pretrained(model_name, 
        debug=False,
        gs_temp_weight=0.01,
        gs_temp_edge=1.0,
        use_weight_masks=False,
        use_edge_masks=False,
    ).to('cuda')
    model.turn_on_weight_masks(deterministic=True, reverse=False)
    model.turn_on_edge_masks(deterministic=True, reverse=False)

    tl_model = HookedTransformer.from_pretrained(model_name).to('cuda')

    res1 = model(tt)[0]
    res2 = tl_model(tt)

    assert torch.allclose(res1, res2, atol=1e-5)