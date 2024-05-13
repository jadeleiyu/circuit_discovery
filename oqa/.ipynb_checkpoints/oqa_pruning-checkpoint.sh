#!/bin/bash
# weight-then-edge
# python oqa_pruning.py --use_weight_masks --prune_edge --resume_epoch_w 114 --train_epochs_edge 100 \
#                       --lambda_sparse_edge_init 10 \
#                       --train_epochs_edge 200 \
#                       --gs_temp_edge 1.0 \
#                       --lambda_complete_edge_init 0 \
#                       --max_times_lambda_sparse_edge 100 --min_times_lambda_sparse_edge 0.01 \
#                       --n_epoch_warmup_lambda_sparse_edge 100 --n_epoch_cooldown_lambda_sparse_edge 20

                            
# weight-only 
python oqa_pruning.py --use_weight_masks \
                            --logits_w_init 1.0 \
                            --gs_temp_weight 1.0 \
                            --lr_weight 0.1 \
                            --prune_weight \
                            --train_epochs_weight 200 \
                            --lambda_sparse_weight_init 1 \
                            --max_times_lambda_sparse_weight 1000 --min_times_lambda_sparse_weight 1000 \
                            --n_epoch_warmup_lambda_sparse_weight 1000 --n_epoch_cooldown_lambda_sparse_weight 500

# edge-only
# python oqa_pruning.py --prune_edge --train_epochs_edge 100 \
#                       --lambda_sparse_edge_init 1 \
#                       --max_times_lambda_sparse_edge 100 --min_times_lambda_sparse_edge 0.01 \
#                       --n_epoch_warmup_lambda_sparse_edge 25 --n_epoch_cooldown_lambda_sparse_edge 25