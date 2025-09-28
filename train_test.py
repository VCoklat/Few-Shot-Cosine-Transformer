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

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(base_loader, val_loader, model, optimization, num_epoch, params):
    import copy
    import torch.optim as optim

    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_CONF'] = 'expandable_segments'

    if optimization == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        raise ValueError('Unknown optimization type')

    # Scheduler: Reduce LR on Plateau with patience of 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True)

    best_val_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    early_stop_patience = 7

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0
        running_acc = 0
        num_batches = 0

        pbar = tqdm.tqdm(total=len(base_loader))
        for i, (x, _) in enumerate(base_loader):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if x.size(0) > 32:
                chunk_size = 16
                losses = []
                accs = []
                for j in range(0, x.size(0), chunk_size):
                    x_chunk = x[j:j+chunk_size].to(device)
                    optimizer.zero_grad()
                    acc, loss = model.set_forward_loss(x_chunk)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())
                    accs.append(acc)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                loss_val = sum(losses)/len(losses)
                acc_val = sum(accs)/len(accs)
            else:
                x = x.to(device)
                optimizer.zero_grad()
                acc, loss = model.set_forward_loss(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_val = loss.item()
                acc_val = acc

            running_loss += loss_val
            running_acc += acc_val
            num_batches += 1

            pbar.set_description(f'Epoch {epoch+1}/{num_epoch} Loss: {loss_val:.4f} Acc: {acc_val*100:.2f}% Mode: {'Advanced' if getattr(model, 'use_advanced_attention', False) else 'Basic'}')
            pbar.update(1)

        pbar.close()

        val_acc = validate(val_loader, model)
        print(f'Validation Accuracy after epoch {epoch+1}: {val_acc*100:.2f}%')

        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'No improvement for {epochs_no_improve} epochs')

        if epochs_no_improve >= early_stop_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    model.load_state_dict(best_model_wts)
    return model

# Helper validate function inside train_test.py

def validate(val_loader, model):
    model.eval()
    accs = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            acc, _ = model.set_forward_loss(x)
            accs.append(acc)
    return sum(accs)/len(accs) if accs else 0

    max_acc = 0
    
    for epoch in range(num_epoch):
        model.train()
        
        # Memory-optimized training loop
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with tqdm.tqdm(total=len(base_loader)) as pbar:
            for i, (x, _) in enumerate(base_loader):
                # Clear cache before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process in smaller chunks if batch is too large
                if x.size(0) > 32:
                    chunk_size = 16
                    chunk_losses = []
                    chunk_accs = []
                    
                    for j in range(0, x.size(0), chunk_size):
                        x_chunk = x[j:j+chunk_size].to(device)
                        
                        optimizer.zero_grad()
                        acc, loss = model.set_forward_loss(x_chunk)
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        chunk_losses.append(loss.item())
                        chunk_accs.append(acc)
                        
                        # Clear cache after each chunk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    avg_loss = np.mean(chunk_losses)
                    avg_acc = np.mean(chunk_accs)
                else:
                    x = x.to(device)
                    optimizer.zero_grad()
                    acc, loss = model.set_forward_loss(x)
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    avg_loss = loss.item()
                    avg_acc = acc
                
                total_loss += avg_loss
                total_acc += avg_acc
                num_batches += 1
                
                pbar.set_description(
                    f'Epoch {epoch+1}/{num_epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}% | Mode: {"Advanced" if model.use_advanced_attention else "Basic"}')
                pbar.update(1)
        
        # Validation phase with memory optimization
        with torch.no_grad():
            model.eval()
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)
            
            val_acc = validate_model(val_loader, model)
            
            if val_acc > max_acc:
                print(f"Best model! Save... Accuracy: {val_acc:.2f}%")
                max_acc = val_acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)
            
            if (epoch % params.save_freq == 0) or (epoch == num_epoch-1):
                outfile = os.path.join(
                    params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)
        
        print(f"Epoch {epoch+1} - Attention Mode: {'Advanced' if model.use_advanced_attention else 'Basic'}")
        print()

    return model

def validate_model(val_loader, model):
    """Memory-optimized validation function"""
    acc_list = []
    
    with tqdm.tqdm(total=len(val_loader)) as pbar:
        for i, (x, _) in enumerate(val_loader):
            # Clear cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Process in chunks to avoid OOM
            if x.size(0) > 16:
                chunk_size = 8
                chunk_accs = []
                
                for j in range(0, x.size(0), chunk_size):
                    x_chunk = x[j:j+chunk_size].to(device)
                    acc, _ = model.set_forward_loss(x_chunk)
                    chunk_accs.append(acc)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_acc = np.mean(chunk_accs)
            else:
                x = x.to(device)
                avg_acc, _ = model.set_forward_loss(x)
            
            acc_list.append(avg_acc * 100)
            pbar.set_description(f'Val | Acc: {np.mean(acc_list):.2f}%')
            pbar.update(1)
    
    return np.mean(acc_list)

def direct_test(test_loader, model, params):
    """Memory-optimized testing function"""
    acc = []
    
    with tqdm.tqdm(total=len(test_loader)) as pbar:
        for i, (x, _) in enumerate(test_loader):
            # Clear cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Process in smaller chunks to avoid OOM
            if x.size(0) > 16:
                scores_list = []
                chunk_size = 8
                
                for j in range(0, x.size(0), chunk_size):
                    x_chunk = x[j:j+chunk_size].to(device)
                    with torch.no_grad():
                        scores_chunk = model.set_forward(x_chunk)
                        scores_list.append(scores_chunk.cpu())
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                scores = torch.cat(scores_list, dim=0)
            else:
                with torch.no_grad():
                    x = x.to(device)
                    scores = model.set_forward(x)

            pred = scores.data.cpu().numpy().argmax(axis=1)
            y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
            acc.append(np.mean(pred == y)*100)

            pbar.set_description(
                f'Test | Acc: {np.mean(acc):.2f}% | Mode: {"Advanced" if model.use_advanced_attention else "Basic"}')
            pbar.update(1)

    acc_all = np.asarray(acc)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    return acc_mean, acc_std

def analyze_dynamic_weights(model, val_loader):
    """Analyze the learned dynamic weights"""
    # Enable weight recording
    for module in model.modules():
        if isinstance(module, Attention):
            module.record_weights = True

    # Run validation to collect weights
    print("Collecting dynamic weight statistics...")
    with torch.no_grad():
        model.eval()
        with tqdm.tqdm(total=min(50, len(val_loader))) as pbar:  # Limit to 50 batches for analysis
            for i, (x, _) in enumerate(val_loader):
                if i >= 50:  # Limit analysis to save time
                    break
                # Process in small chunks for memory efficiency
                if x.size(0) > 8:
                    chunk_size = 4
                    for j in range(0, x.size(0), chunk_size):
                        x_chunk = x[j:j+chunk_size].to(device)
                        model.set_forward(x_chunk)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    x = x.to(device)
                    model.set_forward(x)
                pbar.update(1)

    # Analyze weights
    print("\n" + "="*60)
    print("DYNAMIC WEIGHT ANALYSIS")
    print("="*60)
    
    for i, module in enumerate(model.modules()):
        if isinstance(module, Attention):
            stats = module.get_weight_stats()
            if stats:
                print(f"\nAttention Block {i} Weight Statistics:")
                print("-" * 40)
                if 'cosine_mean' in stats:  # 3-component format
                    print(f"  Cosine weight:     {stats['cosine_mean']:.4f} ± {stats['cosine_std']:.4f}")
                    print(f"  Covariance weight: {stats['cov_mean']:.4f} ± {stats['cov_std']:.4f}")
                    print(f"  Variance weight:   {stats['var_mean']:.4f} ± {stats['var_std']:.4f}")
                    
                    print("\n  Weight Distribution:")
                    for comp in ['cosine', 'cov', 'var']:
                        print(f"    {comp.capitalize()}:")
                        hist = stats['histogram'][comp]
                        for bin_idx, count in enumerate(hist):
                            if count > 0:  # Only show non-zero bins
                                bin_start = bin_idx/10
                                bin_end = (bin_idx+1)/10
                                print(f"      {bin_start:.1f}-{bin_end:.1f}: {count}")
                else:  # Legacy format
                    print(f"  Mean: {stats['mean']:.4f}")
                    print(f"  Std:  {stats['std']:.4f}")
                    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                    
                    print("\n  Distribution:")
                    for bin_idx, count in enumerate(stats['histogram']):
                        if count > 0:  # Only show non-zero bins
                            bin_start = bin_idx/10
                            bin_end = (bin_idx+1)/10
                            print(f"    {bin_start:.1f}-{bin_end:.1f}: {count}")
            
            # Clear history and disable recording
            module.clear_weight_history()
            module.record_weights = False
    
    print("="*60)

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
    # Enable memory optimization from the start
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    params = parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(params))
    print()

    project_name = "Few-Shot_TransFormer"
    if params.dataset == 'Omniglot': 
        params.n_query = 15

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
        if params.backbone == 'Conv4': 
            params.backbone = 'Conv4S'
        if params.backbone == 'Conv6': 
            params.backbone = 'Conv6S'

    optimization = params.optimization

    if params.method in ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine']:
        few_shot_params = dict(
            n_way=params.n_way, k_shot=params.k_shot, n_query=params.n_query)

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

            model = FewShotTransformer(feature_model, variant=variant, **few_shot_params)

        elif params.method in ['CTX_softmax', 'CTX_cosine']:
            variant = 'cosine' if params.method == 'CTX_cosine' else 'softmax'
            input_dim = 512 if "ResNet" in params.backbone else 64

            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=False) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=False)

            model = CTX(feature_model, variant=variant, input_dim=input_dim, **few_shot_params)

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
        model = train(base_loader, val_loader, model, optimization, params.num_epoch, params)

        # Analyze dynamic weights if using cosine variant
        if hasattr(model, 'ATTN') and model.ATTN.dynamic_weight:
            print("\n===================================")
            print("Dynamic Weight Analysis: ")
            analyze_dynamic_weights(model, val_loader)

        print("===================================")
        print("Test phase: ")

        # Clear CUDA cache to free up memory for testing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        iter_num = params.test_iter
        split = params.split

        if params.dataset == 'cross':
            if split == 'base':
                testfile = configs.data_dir['miniImagenet'] + 'all.json'
            else:
                testfile = configs.data_dir['CUB'] + split + '.json'
        elif params.dataset == 'cross_char':
            if split == 'base':
                testfile = configs.data_dir['Omniglot'] + 'noLatin.json'
            else:
                testfile = configs.data_dir['emnist'] + split + '.json'
        else:
            testfile = configs.data_dir[params.dataset] + split + '.json'

        if params.save_iter != -1:
            modelfile = get_assigned_file(params.checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(params.checkpoint_dir)

        test_datamgr = SetDataManager(
            image_size, n_episode=iter_num, **few_shot_params)
        test_loader = test_datamgr.get_data_loader(testfile, aug=False)

        model = model.to(device)

        if modelfile is not None:
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])

        acc_mean, acc_std = direct_test(test_loader, model, params)

        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))
        
        print(f"Final attention mechanism used: {'Advanced' if model.use_advanced_attention else 'Basic'}")

        if params.wandb:
            wandb.log({
                'Test Acc': acc_mean,
                'Test Std': acc_std,
                'Attention Mode': 'Advanced' if model.use_advanced_attention else 'Basic'
            })

        with open('./record/results.txt', 'a') as f:
            timestamp = params.datetime
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-FETI' if params.FETI and 'ResNet' in params.backbone else ''
            
            if params.backbone == "Conv4SNP":
                params.backbone = "Conv4"
            elif params.backbone == "Conv6SNP":
                params.backbone = "Conv6"
            
            exp_setting = '%s-%s-%s%s-%sw%ss' % (params.dataset, params.backbone,
                                                params.method, aug_str, params.n_way, params.k_shot)
            acc_str = 'Test Acc = %4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std/np.sqrt(iter_num))
            attention_mode = 'Advanced' if model.use_advanced_attention else 'Basic'
            f.write('Time: %s Setting: %s %s (Attention: %s)\n' % (timestamp, exp_setting.ljust(50), acc_str, attention_mode))

        if params.wandb:
            wandb.finish()
