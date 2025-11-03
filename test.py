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
from methods.ProFOCT import ProFOCT
import eval_utils

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set memory management parameters
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def direct_test(test_loader, model, params):
    from sklearn.metrics import f1_score
    
    acc = []
    all_preds = []
    all_labels = []
    iter_num = len(test_loader)
    
    with tqdm.tqdm(total=len(test_loader)) as pbar:
        for i, (x, _) in enumerate(test_loader):
            # Process in smaller chunks if needed
            if x.size(0) > 20:  # If batch is larger than 20
                scores_list = []
                chunk_size = 20
                for j in range(0, x.size(0), chunk_size):
                    x_chunk = x[j:j+chunk_size]
                    with torch.cuda.amp.autocast():  # Use mixed precision
                        scores_chunk = model.set_forward(x_chunk)
                    scores_list.append(scores_chunk.cpu())
                scores = torch.cat(scores_list, dim=0)
            else:
                with torch.cuda.amp.autocast():  # Use mixed precision
                    scores = model.set_forward(x)
                    
            pred = scores.data.cpu().numpy().argmax(axis=1)
            y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
            
            all_preds.extend(pred.tolist())
            all_labels.extend(y.tolist())
            
            acc.append(np.mean(pred == y)*100)
            pbar.set_description(
                'Test       | Acc {:.6f}'.format(np.mean(acc)))
            pbar.update(1)

    acc_all = np.asarray(acc)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    
    # Calculate and display per-class F1 scores
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_f1 = f1_score(all_labels, all_preds, average=None)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\n📊 F1 Score Results:")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nPer-class F1 scores:")
    for i, f1 in enumerate(class_f1):
        print(f"  Class {i}: {f1:.4f}")
    
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

def get_class_names_from_file(data_file, n_way=None):
    """
    Generate class names for few-shot evaluation.
    
    In few-shot learning, each episode randomly samples n_way classes from the dataset,
    and labels are re-mapped to 0, 1, ..., n_way-1 for each episode. Therefore, the
    class names should be generic labels representing these positions, not specific
    class names from the dataset (which vary across episodes).
    """
    try:
        with open(data_file, 'r') as f:
            meta = json.load(f)

        unique_labels = np.unique(meta['image_labels']).tolist()
        total_classes = len(unique_labels)
        
        # In few-shot learning, labels are re-mapped to 0 to n_way-1 for each episode
        # Use generic names that represent the position/way rather than specific classes
        if n_way:
            class_names = [f"Way {i}" for i in range(n_way)]
        else:
            # If n_way is not specified, use all available classes
            class_names = [f"Class {i}" for i in range(total_classes)]
        
        print(f"Dataset has {total_classes} classes total, using {len(class_names)} ways for evaluation")
        return class_names
    except Exception as e:
        print(f"Error extracting class names: {e}")
        return [f"Way {i}" for i in range(n_way)] if n_way else ["Class_0"]

if __name__ == '__main__':
    
    params = parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(params))
    print()
    
    if params.dataset == 'Omniglot': params.n_query = min(params.n_query, 15)   #Omniglot only support maximum 15 samples/category as query

    if params.dataset == "CIFAR":
        image_size = 112 if 'ResNet' in params.backbone else 64
    else:
        image_size = 224 if 'ResNet' in params.backbone else 84

    if params.dataset in ['Omniglot', 'cross_char']:
        if params.backbone == 'Conv4': params.backbone = 'Conv4S'
        if params.backbone == 'Conv6': params.backbone = 'Conv6S'

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


    if params.method in ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine', 'ProFOCT_cosine', 'ProFOCT_softmax']:
       
        seed_func()
        
        few_shot_params = dict(
            n_way=params.n_way, k_shot=params.k_shot, n_query=params.n_query)

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
            
        elif params.method in ['ProFOCT_cosine', 'ProFOCT_softmax']:
            variant = 'cosine' if params.method == 'ProFOCT_cosine' else 'softmax'
            
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=True) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=True)
            
            # ProFOCT-specific parameters
            profoct_params = {
                'vic_alpha': params.vic_alpha,
                'vic_beta': params.vic_beta,
                'vic_gamma': params.vic_gamma,
                'dynamic_vic': bool(params.dynamic_vic),
                'distance_metric': params.distance_metric,
                'use_vic_on_attention': bool(params.use_vic_on_attention)
            }
            
            model = ProFOCT(feature_model, variant=variant, **few_shot_params, **profoct_params)
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
        raise ValueError('Can\'t find save model dir')

    print("===================================")
    print("Test phase: ")
    
        
    if params.save_iter != -1:
        modelfile = get_assigned_file(params.checkpoint_dir, params.save_iter)
    else:
        modelfile = get_best_file(params.checkpoint_dir)
        
    test_datamgr = SetDataManager(
        image_size, n_episode=iter_num,  **few_shot_params)
    test_loader = test_datamgr.get_data_loader(testfile, aug=False)
 
    acc_all = []
   
    model = model.to(device)

    root = os.getcwd()

    if params.save_iter != -1:
        modelfile = get_assigned_file(params.checkpoint_dir, params.save_iter)
    else:
        modelfile = get_best_file(params.checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split
    
    # Check if comprehensive evaluation is requested (default: True)
    comprehensive = getattr(params, 'comprehensive_eval', True)
    
    if comprehensive:
        # Get class names from data file
        class_names = get_class_names_from_file(testfile, params.n_way)
        
        # Use comprehensive evaluation
        results = eval_utils.evaluate(test_loader, model, params.n_way, class_names=class_names, device=device)
        eval_utils.pretty_print(results)
        
        # Extract traditional metrics for compatibility
        acc_mean = results['accuracy'] * 100
        acc_std = np.std([f1 * 100 for f1 in results['class_f1']])
    else:
        # Use standard evaluation
        acc_mean, acc_std = direct_test(test_loader, model, params)
        
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %
            (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))
    
            
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
        
        f.write('Time: %s   Setting: %s %s \n' % (timestamp, exp_setting.ljust(50), acc_str))
