
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
import psutil
import subprocess
try:
    import GPUtil
except ImportError:
    GPUtil = None
import sklearn.metrics as metrics
import torch.nn.functional as F

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

def pretty_print(res):
    """Enhanced pretty printing"""
    if not res:
        print("No results to display!")
        return

    print(f"\n📊 EVALUATION RESULTS:")
    print("=" * 50)
    print(f"🎯 Macro-F1: {res['macro_f1']:.4f}")

    print("\n📈 Per-class F1 scores:")
    for name, f in zip(res["class_names"], res["class_f1"]):
        print(f"  F1 '{name}': {f:.4f}")

    print("\n🔢 Confusion matrix:")
    print(np.array(res["conf_mat"]))

    print(f"\n⏱️ Avg inference time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"💾 Model size: {res['param_count']:.2f} M params")
    print(f"🖥️ GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']}/{res['gpu_mem_total_MB']} MB")
    print(f"🖥️ CPU util: {res['cpu_util']}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")
    print("=" * 50)

def get_class_names_from_file(data_file, n_way=None):
    """Extract class names from JSON data file"""
    try:
        with open(data_file, 'r') as f:
            meta = json.load(f)

        unique_labels = np.unique(meta['image_labels']).tolist()

        if 'class_names' in meta:
            class_names = [meta['class_names'][str(label)] for label in unique_labels]
        else:
            class_names = [f"Class_{label}" for label in unique_labels]

        if n_way and len(class_names) > n_way:
            class_names = class_names[:n_way]

        return class_names
    except Exception as e:
        print(f"Error extracting class names: {e}")
        return [f"Class_{i}" for i in range(n_way)] if n_way else ["Class_0"]

def get_system_metrics():
    """Get current system resource usage"""
    try:
        # CPU metrics
        cpu_util = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        cpu_mem_used_MB = memory.used / (1024**2)
        cpu_mem_total_MB = memory.total / (1024**2)

        # GPU metrics
        gpu_util = 0
        gpu_mem_used_MB = 0
        gpu_mem_total_MB = 0

        if torch.cuda.is_available():
            try:
                if GPUtil is not None:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_util = gpu.load
                        gpu_mem_used_MB = gpu.memoryUsed
                        gpu_mem_total_MB = gpu.memoryTotal
                else:
                    # Fallback to torch if GPUtil not available
                    gpu_mem_used_MB = torch.cuda.memory_allocated() / (1024**2)
                    gpu_mem_total_MB = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            except:
                # Fallback to torch if GPUtil fails
                gpu_mem_used_MB = torch.cuda.memory_allocated() / (1024**2)
                gpu_mem_total_MB = torch.cuda.get_device_properties(0).total_memory / (1024**2)

        return {
            'cpu_util': cpu_util,
            'cpu_mem_used_MB': cpu_mem_used_MB,
            'cpu_mem_total_MB': cpu_mem_total_MB,
            'gpu_util': gpu_util,
            'gpu_mem_used_MB': gpu_mem_used_MB,
            'gpu_mem_total_MB': gpu_mem_total_MB
        }
    except Exception as e:
        print(f"Warning: Could not get system metrics: {e}")
        return {
            'cpu_util': 0,
            'cpu_mem_used_MB': 0,
            'cpu_mem_total_MB': 0,
            'gpu_util': 0,
            'gpu_mem_used_MB': 0,
            'gpu_mem_total_MB': 0
        }

def get_model_params(model):
    """Get model parameter count in millions"""
    try:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return param_count / 1e6  # Convert to millions
    except:
        return 0.0

def evaluate_model_comprehensive(test_loader, model, params, testfile):
    """Comprehensive model evaluation with detailed metrics"""
    print("\n🔍 Starting comprehensive model evaluation...")

    # Get class names
    class_names = get_class_names_from_file(testfile, params.n_way)

    # Initialize tracking variables
    all_predictions = []
    all_true_labels = []
    inference_times = []

    # Get initial system metrics
    system_metrics = get_system_metrics()
    param_count = get_model_params(model)

    model.eval()
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_loader), desc="Evaluating") as pbar:
            for i, (x, _) in enumerate(test_loader):
                start_time = time.time()

                # Clear cache before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Process in smaller chunks to avoid OOM
                if x.size(0) > 16:
                    scores_list = []
                    chunk_size = 8
                    for j in range(0, x.size(0), chunk_size):
                        x_chunk = x[j:j+chunk_size].to(device)
                        scores_chunk = model.set_forward(x_chunk)
                        scores_list.append(scores_chunk.cpu())
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    scores = torch.cat(scores_list, dim=0)
                else:
                    x = x.to(device)
                    scores = model.set_forward(x)

                # Calculate inference time
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Get predictions and true labels
                pred = scores.data.cpu().numpy().argmax(axis=1)
                y_true = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)

                all_predictions.extend(pred.tolist())
                all_true_labels.extend(y_true.tolist())

                pbar.update(1)

    # Calculate comprehensive metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # Accuracy
    accuracy = np.mean(all_predictions == all_true_labels) * 100

    # F1 scores
    f1_scores = metrics.f1_score(all_true_labels, all_predictions, average=None, zero_division=0)
    macro_f1 = metrics.f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)

    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(all_true_labels, all_predictions)

    # Timing metrics
    avg_inference_time = np.mean(inference_times)

    # Final system metrics
    final_system_metrics = get_system_metrics()

    # Compile results
    evaluation_results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_f1': f1_scores.tolist(),
        'class_names': class_names,
        'conf_mat': conf_matrix.tolist(),
        'avg_inf_time': avg_inference_time,
        'param_count': param_count,
        'cpu_util': final_system_metrics['cpu_util'],
        'cpu_mem_used_MB': final_system_metrics['cpu_mem_used_MB'],
        'cpu_mem_total_MB': final_system_metrics['cpu_mem_total_MB'],
        'gpu_util': final_system_metrics['gpu_util'],
        'gpu_mem_used_MB': final_system_metrics['gpu_mem_used_MB'],
        'gpu_mem_total_MB': final_system_metrics['gpu_mem_total_MB']
    }

    return evaluation_results

def train(base_loader, val_loader, model, optimization, num_epoch, params):
    # Memory optimization settings
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
                    f'Epoch {epoch+1}/{num_epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}% | Mode: {"Advanced" if hasattr(model, "use_advanced_attention") and model.use_advanced_attention else "Basic"}')
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

        print(f"Epoch {epoch+1} - Attention Mode: {'Advanced' if hasattr(model, 'use_advanced_attention') and model.use_advanced_attention else 'Basic'}")
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
                f'Test | Acc: {np.mean(acc):.2f}% | Mode: {"Advanced" if hasattr(model, "use_advanced_attention") and model.use_advanced_attention else "Basic"}')
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
            if hasattr(module, 'record_weights'):
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
            if hasattr(module, 'get_weight_stats'):
                stats = module.get_weight_stats()
                if stats:
                    print(f"\nAttention Block {i} Weight Statistics:")
                    print("-" * 40)

                    if 'cosine_mean' in stats:  # 3-component format
                        print(f"  Cosine weight: {stats['cosine_mean']:.4f} ± {stats['cosine_std']:.4f}")
                        print(f"  Covariance weight: {stats['cov_mean']:.4f} ± {stats['cov_std']:.4f}")
                        print(f"  Variance weight: {stats['var_mean']:.4f} ± {stats['var_std']:.4f}")

                        print("\n Weight Distribution:")
                        for comp in ['cosine', 'cov', 'var']:
                            print(f"  {comp.capitalize()}:")
                            hist = stats['histogram'][comp]
                            for bin_idx, count in enumerate(hist):
                                if count > 0:  # Only show non-zero bins
                                    bin_start = bin_idx/10
                                    bin_end = (bin_idx+1)/10
                                    print(f"    {bin_start:.1f}-{bin_end:.1f}: {count}")
                    else:  # Legacy format
                        print(f"  Mean: {stats['mean']:.4f}")
                        print(f"  Std: {stats['std']:.4f}")
                        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

                        print("\n Distribution:")
                        for bin_idx, count in enumerate(stats['histogram']):
                            if count > 0:  # Only show non-zero bins
                                bin_start = bin_idx/10
                                bin_end = (bin_idx+1)/10
                                print(f"    {bin_start:.1f}-{bin_end:.1f}: {count}")

                # Clear history and disable recording
                if hasattr(module, 'clear_weight_history'):
                    module.clear_weight_history()
                if hasattr(module, 'record_weights'):
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
        if hasattr(model, 'ATTN') and hasattr(model.ATTN, 'dynamic_weight') and model.ATTN.dynamic_weight:
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

        # Use comprehensive evaluation
        print("\n🚀 Starting comprehensive evaluation...")
        eval_results = evaluate_model_comprehensive(test_loader, model, params, testfile)

        # Pretty print the results
        pretty_print(eval_results)

        # Also compute traditional metrics for compatibility
        acc_mean, acc_std = direct_test(test_loader, model, params)

        print('\n📊 Traditional Test Results:')
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))
        print(f"Final attention mechanism used: {'Advanced' if hasattr(model, 'use_advanced_attention') and model.use_advanced_attention else 'Basic'}")

        if params.wandb:
            # Log both traditional and comprehensive metrics
            wandb.log({
                'Test Acc': acc_mean,
                'Test Std': acc_std,
                'Comprehensive/Macro_F1': eval_results['macro_f1'],
                'Comprehensive/Avg_Inference_Time_ms': eval_results['avg_inf_time'] * 1000,
                'Comprehensive/Model_Size_M': eval_results['param_count'],
                'Comprehensive/GPU_Util_Percent': eval_results['gpu_util'] * 100,
                'Comprehensive/CPU_Util_Percent': eval_results['cpu_util'],
                'Attention Mode': 'Advanced' if hasattr(model, 'use_advanced_attention') and model.use_advanced_attention else 'Basic'
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
            attention_mode = 'Advanced' if hasattr(model, 'use_advanced_attention') and model.use_advanced_attention else 'Basic'

            # Enhanced logging with comprehensive metrics
            f.write('Time: %s Setting: %s %s (Attention: %s) | Macro-F1: %.4f | Inf-Time: %.1fms | Params: %.2fM\n' % 
                   (timestamp, exp_setting.ljust(50), acc_str, attention_mode, 
                    eval_results['macro_f1'], eval_results['avg_inf_time']*1000, eval_results['param_count']))

        if params.wandb:
            wandb.finish()
