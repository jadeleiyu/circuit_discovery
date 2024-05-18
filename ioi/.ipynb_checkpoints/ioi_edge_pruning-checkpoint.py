from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import transformer_lens
from torch.optim import AdamW
from os.path import join
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import argparse

from circuit_gpt import CircuitGPT, CircuitGPTConfig
from ioi_dataset import IOIDataset, CircuitIOIDataset, prepare_batch_inputs, prepare_ioi_data_for_clm

     

def main(args):

    # path that stores gpt-small weights and gpt tokenizer
    model_path = join(args.model_dir, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # load IOI data
    # Note that there are overlaps between train and test sets, due to the way IOIDataset is constructed (randomly sample N items)
    ds = IOIDataset(prompt_type="ABBA", N=args.n_ioi_data, tokenizer=tokenizer)
    ds_train, ds_test = train_test_split(ds.ioi_prompts, test_size=0.2, random_state=0)
    
    ioi_ds_train = CircuitIOIDataset(prepare_ioi_data_for_clm(ds_train))
    ioi_ds_test = CircuitIOIDataset(prepare_ioi_data_for_clm(ds_test))
    
    train_dl = DataLoader(
        ioi_ds_train,
        batch_size=args.batch_size
    )
    eval_dl = DataLoader(
        ioi_ds_train,
        batch_size=args.batch_size,
        shuffle=False
    )

    # circuit_gpt initialization
    device = torch.device('cuda')    
    gpt_weights = torch.load(join(model_path, 'model_weights.pt')) 
    circuit_gpt_config = CircuitGPTConfig(
        debug=False,
        gs_temp_weight=args.gs_temp_weight,
        gs_temp_edge=args.gs_temp_edge,
        use_weight_masks=False
    )
    circuit_gpt = CircuitGPT(circuit_gpt_config)
    circuit_gpt.load_pretrained_weight(gpt_weights)
    
    # load pretrained mask logits if necessary
    if args.resume_epoch_w > 0:
        weight_mask_logits = torch.load(join(model_path), f'weight_mask_logits_{resume_epoch_w}.pt')
        circuit_gpt.load_pretrained_weight_mask(weight_mask_logits)
    if args.resume_epoch_e > 0:
        edge_mask_logits = torch.load(join(model_path), f'edge_mask_logits_{resume_epoch_e}.pt')
        circuit_gpt.load_pretrained_edge_mask(edge_mask_logits)
        
    # move circuit_gpt to gpu after loading all weights and masks
    circuit_gpt.to(device);

    # evaluation before any weight/edge pruning
    # on ioi task, gpt-small should have near-100% acc.
    eval_acc, weight_density, edge_density = eval_model(
        circuit_gpt, eval_dl, tokenizer, device, 
        use_weight_mask=False, use_edge_mask=True
    )    
    print(
        f"Epoch 0. mean pruned model accuracy: {eval_acc:.2f}," + 
        f"weight density: {weight_density:.4f}," + 
        f"edge density: {edge_density:.4f}"
    )

    # optimizers for weight/edge masks
    # weight_logits = [mask for _, mask in circuit_gpt.mask_logits_dict_weight.items()]
    edge_logits = [mask for _, mask in circuit_gpt.mask_logits_dict_edge.items()]
    # optim_weight = torch.optim.AdamW(weight_logits, lr=args.lr_weight)
    optim_edge = torch.optim.AdamW(edge_logits, lr=args.lr_edge)

    # it takes about 35 mins to run 100 epochs of edge mask training on one A100 GPU with batch_size=32
    for epoch in tqdm(range(args.train_epochs_edge)):
        lambda_sparse_edge = get_lambda_sparse(
            epoch, 
            lambda_0=args.lambda_sparse_edge_init,
            max_times=args.max_times_lambda_sparse, 
            min_times=args.min_times_lambda_sparse, 
            n_epoch_warmup=args.n_epoch_warmup_lambda_sparse,
            n_epoch_cooldown=args.n_epoch_cooldown_lambda_sparse,
        )
        lambda_complete_edge = args.lambda_complete_edge_init
        
        for batch in train_dl:
            batch_inputs = prepare_batch_inputs(batch, tokenizer)
            
            circuit_gpt.turn_on_edge_masks(deterministic=False)
            sparse_loss_edge = circuit_gpt.edge_sparseness_loss()
        
            batch_logits = circuit_gpt(batch_inputs['input_ids'].to(device))[0] 
            faith_loss_edge, _ = compute_faith_loss(batch_logits, batch_inputs) 
            
            circuit_gpt.turn_on_edge_masks(deterministic=False, reverse=True)
            batch_logits = circuit_gpt(batch_inputs['input_ids'].to(device))[0] 
            complete_loss_edge, _ = compute_complete_loss(batch_logits, batch_inputs) 
            
            loss_edge = faith_loss_edge + sparse_loss_edge *  lambda_sparse_edge + complete_loss_edge * lambda_complete_edge
            loss_edge.backward()
            optim_edge.step()
            optim_edge.zero_grad()
            torch.cuda.empty_cache()
    
        eval_acc_pruned, weight_density, edge_density = eval_model(
            circuit_gpt, eval_dl, tokenizer, device, use_weight_mask=False, reverse=False
        )
        eval_acc_complement, _, _ = eval_model(
            circuit_gpt, eval_dl, tokenizer, device, use_weight_mask=False, reverse=True
        )
        print(
            "Epoch {}. discovered circuit accuracy: {:.4f}, complementary circuit accuracy: {:.4f}, weight density: {:.4f}, edge density: {:.4f}".format(
                epoch + 1, eval_acc_pruned, eval_acc_complement, weight_density, edge_density)
        )
    
        # save good edge masks
        if eval_acc_pruned > 0.95 and edge_density < 0.05:
            torch.save(
                circuit_gpt.mask_logits_dict_edge,
                join(args.results_dir, f'mask_logits_dict_edge_ioi_edge_only_{epoch}.pt')
            )
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='/home/leiyu/scratch/circuit-discovery/mask_logits/', type=str)
    parser.add_argument('--data_dir', default='/home/leiyu/projects/def-yangxu/leiyu/circuit-discovery/data/', type=str)
    parser.add_argument('--model_dir', default='/home/leiyu/projects/def-yangxu/leiyu/LMs/', type=str)
    parser.add_argument('--model_name', default='gpt2-small', type=str)
    parser.add_argument('--use_weight_masks', default=False, type=bool)
    
    parser.add_argument('--gs_temp_weight', default=0.01, type=float)
    parser.add_argument('--gs_temp_edge', default=1.0, type=float)
    parser.add_argument('--logits_w_init', default=0.0, type=float)
    parser.add_argument('--logits_e_init', default=0.0, type=float)
    parser.add_argument('--lr_weight', default=0.1, type=float)
    parser.add_argument('--lr_edge', default=0.1, type=float)
    parser.add_argument('--lambda_sparse_weight_init', default=1.0, type=float)
    parser.add_argument('--lambda_sparse_edge_init', default=1.0, type=float)
    parser.add_argument('--lambda_complete_weight_init', default=1.0, type=float)
    parser.add_argument('--lambda_complete_edge_init', default=1.0, type=float)
    parser.add_argument('--max_times_lambda_sparse', default=100., type=float)
    parser.add_argument('--min_times_lambda_sparse', default=0.01, type=float)

    parser.add_argument('--n_ioi_data', default=640, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_epochs_weight', default=500, type=int)
    parser.add_argument('--train_epochs_edge', default=100, type=int)
    parser.add_argument('--n_epoch_warmup_lambda_sparse', default=20, type=int)
    parser.add_argument('--n_epoch_cooldown_lambda_sparse', default=20, type=int)
    parser.add_argument('--resume_epoch_w', default=0, type=int)
    parser.add_argument('--resume_epoch_e', default=0, type=int)
    

    args = parser.parse_args()
    main(args)
