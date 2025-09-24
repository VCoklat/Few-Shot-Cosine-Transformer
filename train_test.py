import glob
import json
import os
import pdb
import pprint
import random
import time
import gc
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
import psutil
import GPUtil
from sklearn.metrics import f1_score, confusion_matrix

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

    max_acc = 0
    for epoch in range(num_epoch):
        model.train()
        model.train_loop(epoch, num_epoch, base_loader,
                        params.wandb, optimizer)

        with torch.no_grad():
            model.eval()
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc = model.val_loop(val_loader, epoch, params.wandb)
            if acc > max_acc:
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)

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
            # Process in smaller chunks to avoid OOM
            if x.size(0) > 16: # If batch is larger than 16
                scores_list = []
                chunk_size = 16
                for j in range(0, x.size(0), chunk_size):
                    x_chunk = x[j:j+chunk_size].to(device)
                    with torch.no_grad(): # Ensure no gradients
                        scores_chunk = model.set_forward(x_chunk)
                        scores_list.append(scores_chunk.cpu())
                    torch.cuda.empty_cache() # Clear cache after each chunk
                scores = torch.cat(scores_list, dim=0)
            else:
                with torch.no_grad(): # Ensure no gradients
                    x = x.to(device)
                    scores = model.set_forward(x)

            pred = scores.data.cpu().numpy().argmax(axis=1)
            y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
            acc.append(np.mean(pred == y)*100)
            pbar.set_description(
                'Test | Acc {:.6f}'.format(np.mean(acc)))
            pbar.update(1)

    acc_all = np.asarray(acc)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    return acc_mean, acc_std

@torch.no_grad()
def evaluate(loader, model, n_way, class_names=None, chunk=16, device="cuda"):
    """
    Returns a dict with:
      macro_f1         — single number
      class_f1         — list aligned with class_names (or indices)
      conf_mat         — square matrix (Python list)
      avg_inf_time     — seconds per episode
      param_count      — model size (M)
      gpu / cpu stats  — utilisation & memory
    """
    model.eval()
    all_true, all_pred, times = [], [], []

    for x, _ in loader:                     # dataset's y is ignored
        t0 = time.time()
        if x.size(0) > chunk:               # prevent OOM
            scores = torch.cat([
                model.set_forward(x[i:i+chunk].to(device)).cpu()
                for i in range(0, x.size(0), chunk)], 0)
        else:
            scores = model.set_forward(x.to(device)).cpu()
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        preds = scores.argmax(1).numpy()          # (n_way*n_query,)
        all_pred.append(preds)

        # fabricate ground-truth labels matching preds length
        num_per_class = len(preds) // n_way       # = n_query
        all_true.append(np.repeat(np.arange(n_way), num_per_class))

        del scores; gc.collect(); torch.cuda.empty_cache()

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    res = dict(
        macro_f1   = float(f1_score(y_true, y_pred, average="macro")),
        class_f1   = f1_score(y_true, y_pred, average=None).tolist(),
        conf_mat   = confusion_matrix(y_true, y_pred).tolist(),
        avg_inf_time = float(np.mean(times)),
        param_count  = sum(p.numel() for p in model.parameters())/1e6
    )

    gpus = GPUtil.getGPUs()
    res.update(
        gpu_mem_used_MB   = sum(g.memoryUsed  for g in gpus) if gpus else 0,
        gpu_mem_total_MB  = sum(g.memoryTotal for g in gpus) if gpus else 0,
        gpu_util          = float(sum(g.load for g in gpus)/len(gpus)) if gpus else 0,
        cpu_util          = psutil.cpu_percent(),
        cpu_mem_used_MB   = psutil.virtual_memory().used  / 1_048_576,
        cpu_mem_total_MB  = psutil.virtual_memory().total / 1_048_576,
        class_names       = class_names or list(range(len(res["class_f1"])))
    )
    return res

def pretty_print(res):
    print(f"\nMacro-F1: {res['macro_f1']:.4f}")
    for name, f in zip(res["class_names"], res["class_f1"]):
        print(f"  F1 '{name}': {f:.4f}")

    print("\nConfusion matrix:\n", np.array(res["conf_mat"]))
    print(f"\nAvg inference time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"Model size: {res['param_count']:.2f} M params")
    print(f"GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']}/{res['gpu_mem_total_MB']} MB")
    print(f"CPU util: {res['cpu_util']}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")

def get_class_names_from_file(data_file, n_way=None):
    """Extract class names from JSON data file"""
    try:
        with open(data_file, 'r') as f:
            meta = json.load(f)

        # Get unique class labels
        unique_labels = np.unique(meta['image_labels']).tolist()

        # If class names are available in the meta data
        if 'class_names' in meta:
            class_names = [meta['class_names'][str(label)] for label in unique_labels]
        else:
            # Use the labels themselves as names
            class_names = [f"Class_{label}" for label in unique_labels]

        # If n_way is specified, limit to that number
        if n_way and len(class_names) > n_way:
            class_names = class_names[:n_way]

        return class_names
    except Exception as e:
        print(f"Error extracting class names: {e}")
        return [f"Class_{i}" for i in range(n_way)] if n_way else ["Class_0"]

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
        with tqdm.tqdm(total=len(val_loader)) as pbar:
            for i, (x, _) in enumerate(val_loader):
                x = x.to(device)
                model.set_forward(x)
                pbar.update(1)

    # Analyze weights
    for i, module in enumerate(model.modules()):
        if isinstance(module, Attention):
            stats = module.get_weight_stats()
            if stats:
                print(f"Attention Block {i} weight stats:")
                if 'cosine_mean' in stats: # 3-component format
                    print(f"  Cosine weight: {stats['cosine_mean']:.4f} ± {stats['cosine_std']:.4f}")
                    print(f"  Covariance weight: {stats['cov_mean']:.4f} ± {stats['cov_std']:.4f}")
                    print(f"  Variance weight: {stats['var_mean']:.4f} ± {stats['var_std']:.4f}")
                    print("  Distribution:")
                    for comp in ['cosine', 'cov', 'var']:
                        print(f"    {comp.capitalize()}:")
                        for bin_idx, count in enumerate(stats['histogram'][comp]):
                            bin_start = bin_idx/10
                            bin_end = (bin_idx+1)/10
                            print(f"      {bin_start:.1f}-{bin_end:.1f}: {count}")
                else: # Legacy format
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

    if params.method in ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine']:
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

        # Call this function after training
        analyze_dynamic_weights(model)

        ######################################################################
        print("===================================")
        print("Test phase: ")

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

        # Implement memory optimization
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

        # Get class names for the test dataset
        try:
            class_names = get_class_names_from_file(testfile, params.n_way)
            print(f"Using class names: {class_names}")
        except:
            # Fallback to generic names
            class_names = [f"Class_{i}" for i in range(params.n_way)]
            print(f"Using generic class names: {class_names}")

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

        split = params.split
        if params.save_iter != -1:
            split_str = split + "_" + str(params.save_iter)
        else:
            split_str = split

        # Original accuracy test
        print("\n=== Standard Accuracy Test ===")
        acc_mean, acc_std = direct_test(test_loader, model, params)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))

        # Enhanced metric evaluation
        print("\n=== Detailed Metric Assessment ===")
        res = evaluate(test_loader, model, params.n_way, class_names=class_names, device=device)
        pretty_print(res)

        if params.wandb:
            wandb.log({
                'Test Acc': acc_mean,
                'Macro F1': res['macro_f1'],
                'Avg Inference Time (ms)': res['avg_inf_time'] * 1000,
                'Model Size (M)': res['param_count']
            })

            # Log per-class F1 scores
            for i, (name, f1) in enumerate(zip(res['class_names'], res['class_f1'])):
                wandb.log({f'F1_Class_{name}': f1})

        # Save results to file
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

            acc_str = 'Test Acc = %4.2f%% +- %4.2f%% | Macro F1 = %4.4f' % (
                acc_mean, 1.96 * acc_std/np.sqrt(iter_num), res['macro_f1'])

            f.write('Time: %s Setting: %s %s \n' % (timestamp, exp_setting.ljust(50), acc_str))

            # Write detailed per-class F1 scores
            f.write('Per-class F1 scores: ')
            for name, f1 in zip(res['class_names'], res['class_f1']):
                f.write(f'{name}={f1:.4f} ')
            f.write('\n')

        if params.wandb:
            wandb.finish()
