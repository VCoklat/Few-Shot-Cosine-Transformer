import glob
import json
import os
import pdb
import pprint
import random
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data.sampler
import tqdm
from torch.autograd import Variable
from torchsummary import summary

import backbone
import configs
import data.feature_loader as feat_loader
import wandb
from data.datamgr import SetDataManager
from io_utils import (get_assigned_file, get_best_file,
                      model_dict, parse_args)
from methods.CTX import CTX
from methods.transformer import FewShotTransformer
from methods.transformer import Attention
from methods.fsct_profonet import FSCT_ProFONet

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(base_loader, val_loader, model, optimization, num_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=params.learning_rate * 0.01
    )
    
    # Add warmup scheduler for first few epochs
    warmup_epochs = min(5, num_epoch // 10)  # Warmup for first 5 epochs or 10% of training
    
    max_acc = 0
    # Early stopping parameters to prevent overfitting
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0
    min_delta = 0.1  # Minimum improvement to reset patience counter

    for epoch in range(num_epoch):
        model.train()
        
        # Apply learning rate warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = params.learning_rate * warmup_factor

        model.train_loop(epoch, num_epoch, base_loader,
                         params.wandb,  optimizer)
        
        # Step the scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()
            
        with torch.no_grad():
            model.eval()

            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc = model.val_loop(val_loader, epoch, params.wandb)
            
            # Early stopping logic
            if acc > max_acc + min_delta:  
                print("best model! save...")
                max_acc = acc
                patience_counter = 0  # Reset patience counter
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)
                # if params.wandb:
                #     wandb.save(outfile)
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epoch(s). Best val acc: {max_acc:.2f}%")
                
            # Check if we should stop early
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs. Best val acc: {max_acc:.2f}%")
                break

            if (epoch % params.save_freq == 0) or (epoch == num_epoch-1):
                outfile = os.path.join(
                    params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)
        print()

    return model

def direct_test(test_loader, model, params):

    correct = 0
    count = 0
    acc = []

    iter_num = len(test_loader)
    with tqdm.tqdm(total=len(test_loader)) as pbar:
        for i, (x, _) in enumerate(test_loader):
            scores = model.set_forward(x)
            pred = scores.data.cpu().numpy().argmax(axis=1)
            y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
            acc.append(np.mean(pred == y)*100)
            pbar.set_description(
                'Test       | Acc {:.6f}'.format(np.mean(acc)))
            pbar.update(1)

    acc_all = np.asarray(acc)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    return acc_mean, acc_std

def seed_func():
    seed = 4040 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(10)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def change_model(model_name):
    if model_name == 'Conv4':
        model_name = 'Conv4NP'
    elif model_name == 'Conv6':
        model_name = 'Conv6NP'
    elif model_name == 'Conv4S':
        model_name = 'Conv4SNP'
    elif model_name == 'Conv6S':
        model_name = 'Conv6SNP'
    return model_name

if __name__ == '__main__':
    
    params = parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(params))
    print()
        
    project_name = "Few-Shot_TransFormer"            
    
    if params.dataset == 'Omniglot': params.n_query = 15
    if params.wandb:

        wandb_name = params.method + "_" + params.backbone + "_" + params.dataset + \
            "_" + str(params.n_way) + "w" + str(params.k_shot) + "s"
        
        if params.train_aug:
            wandb_name += "_aug"
        if params.FETI and 'ResNet' in params.backbone:
            wandb_name += "_FETI"
        wandb_name += "_" + params.datetime
        
        wandb.init(project=project_name, name=wandb_name,
                   config=params, id=params.datetime)
        print()

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['Omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if params.dataset == "CIFAR":
        image_size = 112 if 'ResNet' in params.backbone else 64
    else:
        image_size = 224 if 'ResNet' in params.backbone else 84

    if params.dataset in ['Omniglot', 'cross_char']:
        if params.backbone == 'Conv4': params.backbone = 'Conv4S'
        if params.backbone == 'Conv6': params.backbone = 'Conv6S'

    optimization = params.optimization

    if params.method in ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine', 'FSCT_ProFONet']:

        few_shot_params = dict(
            n_way=params.n_way, k_shot=params.k_shot, n_query = params.n_query)
        base_datamgr = SetDataManager(
            image_size, n_episode=params.n_episode, **few_shot_params)
        base_loader = base_datamgr.get_data_loader(
            base_file, aug=params.train_aug)

        val_datamgr = SetDataManager(
            image_size, n_episode=params.n_episode, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(
            val_file, aug=False)
       
        seed_func()

        if params.method in ['FSCT_softmax', 'FSCT_cosine']:
            variant = 'cosine' if params.method == 'FSCT_cosine' else 'softmax'
            
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=True) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=True)

            # Balanced hyperparameters to reduce overfitting (train 97% -> val 60%)
            # Reduced model capacity and increased regularization for better generalization
            model = FewShotTransformer(
                feature_model, 
                variant=variant, 
                depth=1,  # Reduced from 2 to 1 to prevent overfitting
                heads=8,  # Reduced from 12 to 8 for simpler attention
                dim_head=64,  # Reduced from 80 to 64 to decrease capacity
                mlp_dim=512,  # Reduced from 768 to 512 to prevent memorization
                initial_cov_weight=0.55,  # Optimized covariance weight
                initial_var_weight=0.2,  # Optimized variance weight
                dynamic_weight=True,  # Enable dynamic weighting
                label_smoothing=0.15,  # Increased from 0.1 to 0.15 for stronger regularization
                attention_dropout=0.2,  # Increased from 0.15 to 0.2 for better generalization
                drop_path_rate=0.15,  # Increased from 0.1 to 0.15 for stronger regularization
                **few_shot_params
            )
            
        elif params.method in ['CTX_softmax', 'CTX_cosine']:
            variant = 'cosine' if params.method == 'CTX_cosine' else 'softmax'
            input_dim = 512 if "ResNet" in params.backbone else 64
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=False) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=False)

            model = CTX(feature_model, variant=variant, input_dim=input_dim, **few_shot_params)
        
        elif params.method == 'FSCT_ProFONet':
            # Hybrid FS-CT + ProFONet method
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=True) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=True)
            
            # Use optimized parameters for 8GB VRAM
            model = FSCT_ProFONet(
                feature_model,
                variant='cosine',
                depth=1,
                heads=4,
                dim_head=160,
                mlp_dim=512,
                dropout=0.0,
                lambda_V_base=0.5,
                lambda_I=9.0,
                lambda_C_base=0.5,
                gradient_checkpointing=True if torch.cuda.is_available() else False,
                mixed_precision=True if torch.cuda.is_available() else False,
                **few_shot_params
            )
    else:
        raise ValueError('Unknown method')

    model = model.to(device)
    
    
    params.checkpoint_dir = '%sc/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.backbone, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if params.FETI and 'ResNet' in params.backbone:
        params.checkpoint_dir += '_FETI'

    params.checkpoint_dir += '_%dway_%dshot' % (
        params.n_way, params.k_shot)
    
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    
    print("===================================")
    print("Train phase: ")
    
    model = train(base_loader, val_loader,  model, optimization, params.num_epoch, params)

    # Add to train.py after training loop
    def analyze_dynamic_weights(model):
        """Analyze the learned dynamic weights"""
        # Enable weight recording
        for module in model.modules():
            if isinstance(module, Attention):
                module.record_weights = True
        
        # Run validation to collect weights
        print("Collecting dynamic weight statistics...")
        with torch.no_grad():
            model.eval()
            model.val_loop(val_loader, 0, False)
        
        # Analyze weights
        for i, module in enumerate(model.modules()):
            if isinstance(module, Attention):
                stats = module.get_weight_stats()
                if stats:
                    print(f"Attention Block {i} weight stats:")
                    print(f"  Mean: {stats['mean']:.4f}")
                    print(f"  Std: {stats['std']:.4f}")
                    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                    print("  Distribution:")
                    for bin_idx, count in enumerate(stats['histogram']):
                        bin_start = bin_idx/10
                        bin_end = (bin_idx+1)/10
                        print(f"    {bin_start:.1f}-{bin_end:.1f}: {count}")
                module.clear_weight_history()
                module.record_weights = False

    # Call this function after training
    analyze_dynamic_weights(model)

######################################################################

    print("===================================")
    print("Test phase: ")

