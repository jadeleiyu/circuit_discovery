#!/bin/bash
# weight-then-edge
# python ioi_pruning.py --use_weight_masks --prune_edge --ds_idx 0 --resume_epoch_w 257 --train_epochs_edge 100 \
#                       --lambda_sparse_edge_init 10 \
#                       --max_times_lambda_sparse_edge 100 --min_times_lambda_sparse_edge 0.01 \
#                       --n_epoch_warmup_lambda_sparse_edge 10 --n_epoch_cooldown_lambda_sparse_edge 20


# weight-only
# python ioi_pruning.py --use_weight_masks \
#                             --prune_weight \
#                             --ds_idx 0 \
#                             --train_epochs_weight 300 


# edge-only
python ioi_pruning.py --prune_edge --ds_idx 2 --train_epochs_edge 100 \
                      --lambda_sparse_edge_init 1 \
                      --max_times_lambda_sparse_edge 100 --min_times_lambda_sparse_edge 0.01 \
                      --n_epoch_warmup_lambda_sparse_edge 50 --n_epoch_cooldown_lambda_sparse_edge 50