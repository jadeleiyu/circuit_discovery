import argparse
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--model-weight-dir-path', default='/home/t-jniu/tl_model_weights', type=Path)

    args = parser.parse_args()

    model_path = args.model_weight_dir_path / Path(args.model_name).stem
    if model_path.exists():
        raise ValueError

    model = HookedTransformer.from_pretrained(args.model_name)
    model_path.mkdir(parents=True)
    torch.save(model.state_dict(), model_path / 'model_weights.pt')