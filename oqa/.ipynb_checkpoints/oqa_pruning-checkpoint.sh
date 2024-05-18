#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --array=0-20
#SBATCH --account=def-yangxu
#SBATCH --output=./slurm_out/%A_%a.out
#SBATCH --error=./slurm_out/%A_%a.err

# weight-then-edge
python oqa_pruning.py --exp_name weight-then-edge  --use_weight_masks \
                      --prune_weight \
                      --train_epochs_weight 100 \
                      --lambda_sparse_weight_init 1 \
                      --gs_temp_weight 1.0 \
                      --max_times_lambda_sparse_weight 1000 --min_times_lambda_sparse_weight 1000 \
                      --n_epoch_warmup_lambda_sparse_weight 1000 --n_epoch_cooldown_lambda_sparse_weight 500 \
                      --prune_edge \
                      --train_epochs_edge 20 \
                      --lambda_sparse_edge_init 10 \
                      --gs_temp_edge 1.0 \
                      --lambda_complete_edge_init 0 \
                      --max_times_lambda_sparse_edge 100 --min_times_lambda_sparse_edge 0.01 \
                      --n_epoch_warmup_lambda_sparse_edge 100 --n_epoch_cooldown_lambda_sparse_edge 20 \
                      --job_id $SLURM_ARRAY_TASK_ID 


                            
# weight-only 
# python oqa_pruning.py  --use_weight_masks \
#                             --logits_w_init 1.0 \
#                             --gs_temp_weight 1.0 \
#                             --lr_weight 0.1 \
#                             --prune_weight \
#                             --train_epochs_weight 110 \
#                             --lambda_sparse_weight_init 1 \
#                             --max_times_lambda_sparse_weight 1000 --min_times_lambda_sparse_weight 1000 \
#                             --n_epoch_warmup_lambda_sparse_weight 1000 --n_epoch_cooldown_lambda_sparse_weight 500 \
#                             --exp_name weight-only \
#                             --job_id $SLURM_ARRAY_TASK_ID 

# edge-only
# python oqa_pruning.py --prune_edge --train_epochs_edge 100 \
#                       --lambda_sparse_edge_init 1 \
#                       --max_times_lambda_sparse_edge 100 --min_times_lambda_sparse_edge 0.01 \
#                       --n_epoch_warmup_lambda_sparse_edge 25 --n_epoch_cooldown_lambda_sparse_edge 25