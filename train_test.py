
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

# Additional imports for enhanced functionality
try:
    import psutil
except ImportError:
    print("psutil not installed. System monitoring will be limited.")
    psutil = None

try:
    import GPUtil
except ImportError:
    print("GPUtil not installed. GPU monitoring will be limited.")
    GPUtil = None

from sklearn.metrics import confusion_matrix, f1_score

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
import eval_utils

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
    for i, f in enumerate(res["class_f1"]):
        name = res["class_names"][i] if i < len(res["class_names"]) else f"Class {i}"
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
                        result = model.set_forward(x_chunk)
                        # Handle tuple return (e.g., FSCT_ProFONet returns (scores, z_support, z_proto))
                        scores_chunk = result[0] if isinstance(result, tuple) else result
                        scores_list.append(scores_chunk.cpu())
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    scores = torch.cat(scores_list, dim=0)
                else:
                    x = x.to(device)
                    result = model.set_forward(x)
                    # Handle tuple return (e.g., FSCT_ProFONet returns (scores, z_support, z_proto))
                    scores = result[0] if isinstance(result, tuple) else result

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

def safe_checkpoint_save(checkpoint_dict, filepath, max_retries=3):
    """
    Safely save a checkpoint with error handling and cleanup.
    
    Args:
        checkpoint_dict: Dictionary containing model state to save
        filepath: Path where checkpoint should be saved
        max_retries: Maximum number of retry attempts
    
    Returns:
        bool: True if save succeeded, False otherwise
    """
    import shutil
    import tempfile
    
    # Clean up old checkpoint files to free space (keep only last 3)
    checkpoint_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # Only clean up numbered checkpoints, not best_model.tar
    if filename != 'best_model.tar' and filename.endswith('.tar'):
        try:
            # Get all numbered checkpoint files
            checkpoint_files = []
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.tar') and f != 'best_model.tar':
                    try:
                        # Extract epoch number from filename
                        epoch_num = int(f.replace('.tar', ''))
                        checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, f)))
                    except ValueError:
                        continue
            
            # Sort by epoch number and keep only the last 2 (plus the one we're about to save = 3 total)
            if len(checkpoint_files) > 2:
                checkpoint_files.sort(key=lambda x: x[0])
                # Remove older checkpoints
                for epoch_num, old_file in checkpoint_files[:-2]:
                    try:
                        os.remove(old_file)
                        print(f"Removed old checkpoint: {os.path.basename(old_file)}")
                    except Exception as e:
                        print(f"Warning: Could not remove old checkpoint {old_file}: {e}")
        except Exception as e:
            print(f"Warning: Error during checkpoint cleanup: {e}")
    
    # Try to save with retries
    for attempt in range(max_retries):
        try:
            # Use atomic write pattern: write to temp file first, then rename
            temp_fd, temp_path = tempfile.mkstemp(dir=checkpoint_dir, suffix='.tar.tmp')
            os.close(temp_fd)  # Close the file descriptor
            
            try:
                # Save to temporary file
                torch.save(checkpoint_dict, temp_path)
                
                # Atomic rename
                shutil.move(temp_path, filepath)
                
                print(f"✓ Checkpoint saved successfully: {os.path.basename(filepath)}")
                return True
                
            except Exception as e:
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise e
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Warning: Checkpoint save attempt {attempt + 1} failed: {e}")
                print(f"Retrying... ({attempt + 2}/{max_retries})")
                time.sleep(1)  # Brief pause before retry
            else:
                print(f"✗ ERROR: Failed to save checkpoint after {max_retries} attempts")
                print(f"  File: {filepath}")
                print(f"  Error: {e}")
                print(f"  Training will continue without saving this checkpoint.")
                return False
    
    return False

def validate_model(val_loader, model):
    """
    Validate the model on the validation set.
    
    Args:
        val_loader: DataLoader for validation data
        model: The model to validate
        
    Returns:
        float: Validation accuracy percentage
    """
    correct = 0
    count = 0
    acc_all = []
    
    model.eval()
    with torch.no_grad():
        iter_num = len(val_loader)
        with tqdm.tqdm(total=iter_num, desc="Validation") as val_pbar:
            for i, (x, _) in enumerate(val_loader):
                # Handle dynamic way changes if model supports it
                if hasattr(model, 'change_way') and model.change_way:
                    model.n_way = x.size(0)
                
                # Get predictions
                correct_this, count_this = model.correct(x)
                acc_all.append(correct_this / count_this * 100)
                
                val_pbar.set_description('Validation | Acc {:.2f}%'.format(np.mean(acc_all)))
                val_pbar.update(1)
    
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    
    print('Val Acc = %4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
    
    return acc_mean

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

    # Enable mixed precision training for better memory efficiency
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Add learning rate scheduler for better convergence
    # CosineAnnealingLR helps the model converge better by gradually reducing learning rate
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-6)
    
    # Gradient accumulation steps to reduce memory usage
    accumulation_steps = 2

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
                        
                        # Use mixed precision for memory efficiency
                        if scaler is not None:
                            with torch.cuda.amp.autocast():
                                acc, loss = model.set_forward_loss(x_chunk)
                            # Scale loss for gradient accumulation
                            loss = loss / accumulation_steps
                            scaler.scale(loss).backward()
                        else:
                            acc, loss = model.set_forward_loss(x_chunk)
                            loss = loss / accumulation_steps
                            loss.backward()
                        
                        chunk_losses.append(loss.item() * accumulation_steps)
                        chunk_accs.append(acc)
                        
                        # Update weights after accumulation steps
                        if (j // chunk_size + 1) % accumulation_steps == 0:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                            optimizer.zero_grad()
                        
                        # Clear cache after each chunk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    avg_loss = np.mean(chunk_losses)
                    avg_acc = np.mean(chunk_accs)
                else:
                    x = x.to(device)
                    
                    # Use mixed precision for memory efficiency
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            acc, loss = model.set_forward_loss(x)
                        # Scale loss for gradient accumulation
                        loss = loss / accumulation_steps
                        scaler.scale(loss).backward()
                    else:
                        acc, loss = model.set_forward_loss(x)
                        loss = loss / accumulation_steps
                        loss.backward()
                    
                    # Update weights after accumulation steps
                    if (i + 1) % accumulation_steps == 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                        optimizer.zero_grad()
                    
                    avg_loss = loss.item() * accumulation_steps
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

            # FIXED: Changed from acc > 40% to val_acc > 40%
            if val_acc > 40:
                print(f"🎯 Validation accuracy above 40%! Current: {val_acc:.2f}%")

            if val_acc > max_acc:
                print(f"Best model! Save... Accuracy: {val_acc:.2f}%")
                max_acc = val_acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                safe_checkpoint_save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)

            if (epoch % params.save_freq == 0) or (epoch == num_epoch-1):
                outfile = os.path.join(
                    params.checkpoint_dir, '{:d}.tar'.format(epoch))
                safe_checkpoint_save(
                    {'epoch': epoch, 'state': model.state_dict()}, outfile)
        
        # Step the learning rate scheduler
        scheduler.step()

        print(f"Epoch {epoch+1} - Attention Mode: {'Advanced' if hasattr(model, 'use_advanced_attention') and model.use_advanced_attention else 'Basic'} - LR: {scheduler.get_last_lr()[0]:.6f}")
        print()
    return model

def direct_test(test_loader, model, params, data_file=None, comprehensive=True):
    """Enhanced testing function with optional comprehensive evaluation"""
    if comprehensive and data_file:
        # Get class names from data file
        class_names = get_class_names_from_file(data_file, params.n_way)
        
        # Use eval_utils comprehensive evaluation
        results = eval_utils.evaluate(test_loader, model, params.n_way, class_names=class_names, device=device)
        eval_utils.pretty_print(results)
        
        # Still return traditional accuracy metrics for backward compatibility
        acc_mean = results['accuracy'] * 100  # Convert accuracy to percentage scale
        acc_std = np.std([f1 * 100 for f1 in results['class_f1']])  # Std of class F1s
        
        return acc_mean, acc_std, results
    else:
        # Original direct_test functionality with F1 score calculation
        acc = []
        all_preds = []
        all_labels = []
        iter_num = len(test_loader)
        
        if iter_num == 0:
            print("ERROR: Test loader is empty")
            return 0.0, 0.0

        with tqdm.tqdm(total=len(test_loader)) as pbar:
            for i, (x, _) in enumerate(test_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    result = model.set_forward(x)
                    # Handle tuple return (e.g., FSCT_ProFONet returns (scores, z_support, z_proto))
                    scores = result[0] if isinstance(result, tuple) else result
                    pred = scores.data.cpu().numpy().argmax(axis=1)
                    
                    # Move computation to CPU and free GPU memory
                    y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
                    
                    all_preds.extend(pred.tolist())
                    all_labels.extend(y.tolist())
                    
                    acc.append(np.mean(pred == y)*100)
                    
                    # Clear unnecessary tensors
                    del scores
                    torch.cuda.empty_cache()
                    
                    pbar.set_description('Test | Acc {:.6f}'.format(np.mean(acc)))
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

    if params.method in ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine', 'FSCT_ProFONet']:
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

            # Enable dynamic weighting for improved accuracy
            # Optimized initial weights: higher covariance weight helps with feature separation
            model = FewShotTransformer(feature_model, variant=variant, 
                                     initial_cov_weight=0.5,  # Increased for stronger covariance regularization
                                     initial_var_weight=0.25, # Balanced variance regularization
                                     dynamic_weight=True,
                                     **few_shot_params)

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
