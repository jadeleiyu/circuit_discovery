#!/bin/bash
# python agreement_pruning.py --use_weight_masks \
#                             --prune_weight \
#                             --prune_edge \
#                             --task_name anaphor_gender_agreement

python agreement_pruning.py --use_weight_masks \
                            --prune_edge \
                            --task_name anaphor_gender_agreement \
                            --train_epochs_edge 100 \
                            --resume_epoch_w 210