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
    all_preds_global = []
    all_labels_global = []
    iter_num = len(test_loader)
    
    # Get batch sampler and dataset for tracking all classes
    batch_sampler = test_loader.batch_sampler if hasattr(test_loader, 'batch_sampler') else None
    dataset = test_loader.dataset if hasattr(test_loader, 'dataset') else None
    episode_iter = iter(batch_sampler) if batch_sampler else None
    
    with tqdm.tqdm(total=len(test_loader)) as pbar:
        for i, (x, _) in enumerate(test_loader):
            # Process in smaller chunks if needed - more conservative to prevent OOM
            if x.size(0) > 16:  # Reduced from 20 to 16
                scores_list = []
                chunk_size = 8  # Reduced from 20 to 8
                for j in range(0, x.size(0), chunk_size):
                    x_chunk = x[j:j+chunk_size]
                    with torch.cuda.amp.autocast():  # Use mixed precision
                        scores_chunk = model.set_forward(x_chunk)
                    scores_list.append(scores_chunk.cpu())
                    # Clear cache after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                scores = torch.cat(scores_list, dim=0)
            else:
                with torch.cuda.amp.autocast():  # Use mixed precision
                    scores = model.set_forward(x)
                    
            pred = scores.data.cpu().numpy().argmax(axis=1)
            y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
            
            all_preds.extend(pred.tolist())
            all_labels.extend(y.tolist())
            
            # Track actual class labels
            if episode_iter and dataset:
                try:
                    sampled_class_indices = next(episode_iter)
                    actual_class_ids = [dataset.cl_list[idx] for idx in sampled_class_indices]
                    
                    # Map predictions and labels to actual class IDs
                    n_query = len(pred) // params.n_way
                    pred_global = [actual_class_ids[p] for p in pred]
                    true_global = np.repeat(actual_class_ids, n_query).tolist()
                    
                    all_preds_global.extend(pred_global)
                    all_labels_global.extend(true_global)
                except (StopIteration, AttributeError, IndexError):
                    episode_iter = None
            
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
    class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(params.n_way)), zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', labels=list(range(params.n_way)), zero_division=0)
    
    print(f"\n📊 F1 Score Results:")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nPer-class F1 scores:")
    for i, f1 in enumerate(class_f1):
        print(f"  Class {i}: {f1:.4f}")
    
    # Display all-classes F1 scores if we tracked them
    if all_preds_global and all_labels_global:
        all_preds_global = np.array(all_preds_global)
        all_labels_global = np.array(all_labels_global)
        
        all_class_ids = np.unique(all_labels_global)
        all_classes_f1 = f1_score(all_labels_global, all_preds_global, average=None, 
                                   labels=all_class_ids, zero_division=0)
        
        # Get class names if available
        if dataset and hasattr(dataset, 'class_labels'):
            # Map class IDs directly to class names (cls_id is the index in class_labels)
            all_classes_names = []
            for cls_id in all_class_ids:
                try:
                    # cls_id is the direct index into class_labels array
                    all_classes_names.append(dataset.class_labels[cls_id])
                except (IndexError):
                    all_classes_names.append(f"Class {cls_id}")
        else:
            all_classes_names = [f"Class {cls_id}" for cls_id in all_class_ids]
        
        print(f"\n📊 F1 Scores for All Dataset Classes ({len(all_classes_f1)} classes):")
        for name, f1 in zip(all_classes_names, all_classes_f1):
            print(f"  {name}: {f1:.4f}")
    
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

def get_class_names_from_file(data_file, n_way=None, for_episodes=True):
    """
    Generate class names for few-shot evaluation.
    
    Args:
        data_file: Path to the dataset JSON file
        n_way: Number of ways in few-shot evaluation
        for_episodes: If True, returns generic "Way X" labels for episodic evaluation.
                     If False, returns all actual class names from dataset.
    
    In few-shot learning, each episode randomly samples n_way classes from the dataset,
    and labels are re-mapped to 0, 1, ..., n_way-1 for each episode. Therefore, for
    episodic evaluation we use generic "Way X" labels. For per-class F1 scores across
    all episodes, we use actual class names.
    """
    try:
        with open(data_file, 'r') as f:
            meta = json.load(f)

        unique_labels = np.unique(meta['image_labels']).tolist()
        total_classes = len(unique_labels)
        
        if for_episodes:
            # For episodic evaluation: use generic way labels
            if n_way:
                class_names = [f"Way {i}" for i in range(n_way)]
            else:
                class_names = [f"Class {i}" for i in range(total_classes)]
            print(f"Dataset has {total_classes} classes total, using {len(class_names)} ways for episodic evaluation")
        else:
            # For all-classes evaluation: use actual class names if available
            if 'label_names' in meta:
                class_names = meta['label_names']
            else:
                class_names = [f"Class {i}" for i in range(total_classes)]
            print(f"Dataset has {total_classes} classes total")
        
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


    if params.method in ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine']:
       
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
    feature_analysis = getattr(params, 'feature_analysis', True)
    
    if comprehensive:
        # Get class names from data file for episodic evaluation (Way 0, Way 1, etc.)
        class_names = get_class_names_from_file(testfile, params.n_way, for_episodes=True)
        
        # Use comprehensive evaluation with optional feature analysis
        results = eval_utils.evaluate_comprehensive(
            test_loader, model, params.n_way, 
            class_names=class_names, 
            device=device,
            feature_analysis=feature_analysis
        )
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
