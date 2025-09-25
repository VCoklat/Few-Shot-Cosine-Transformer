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

# TIER 3+ ADVANCED: Enhanced monitoring function
def monitor_advanced_features(model, test_loader, device='cuda', epoch=0, max_episodes=5):
    """Monitor dynamic weights and complex formulas with detailed analysis"""
    model.eval()
    print(f"\n🔍 TIER 3+ ADVANCED MONITORING - Epoch {epoch}")
    print("="*60)

    # Enable weight recording for dynamic weight analysis
    for module in model.modules():
        if hasattr(module, 'record_weights'):
            module.record_weights = True

    total_score_vars = []
    component_stats = {'cosine': [], 'cov': [], 'var': []}

    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= max_episodes:
                break

            x = x.to(device)
            scores = model.set_forward(x)

            # Critical variance monitoring
            score_var = torch.var(scores).item()
            score_std = torch.std(scores).item()
            score_range = (scores.max() - scores.min()).item()
            total_score_vars.append(score_var)

            predictions = torch.argmax(scores, dim=1)
            unique_preds = len(torch.unique(predictions))

            print(f"Episode {i+1}:")
            print(f"  📊 Score variance: {score_var:.6f} (target: >0.01)")
            print(f"  📈 Score std: {score_std:.6f} (target: >0.1)")  
            print(f"  📏 Score range: [{scores.min():.3f}, {scores.max():.3f}] (separation: {score_range:.3f})")
            print(f"  🔢 Unique predictions: {unique_preds}/5")

            # Success indicators
            breakthrough_indicators = []
            if score_var > 0.01:
                breakthrough_indicators.append("✅ VARIANCE BREAKTHROUGH!")
            if unique_preds >= 4:
                breakthrough_indicators.append("✅ MULTI-CLASS SUCCESS!")
            if score_range > 1.0:
                breakthrough_indicators.append("✅ GOOD SEPARATION!")
            if score_std > 0.1:
                breakthrough_indicators.append("✅ HEALTHY DIVERSITY!")

            if breakthrough_indicators:
                for indicator in breakthrough_indicators:
                    print(f"  {indicator}")
            else:
                print(f"  ⚠️  Still need improvement...")

    # Analyze dynamic weights in detail
    print(f"\n🎛️  DYNAMIC WEIGHT ANALYSIS:")
    weight_learning_detected = False
    for i, module in enumerate(model.modules()):
        if hasattr(module, 'get_weight_stats') and hasattr(module, 'dynamic_weight'):
            if module.dynamic_weight:
                stats = module.get_weight_stats()
                if stats and 'cosine_mean' in stats:
                    print(f"  Attention Block {i}:")
                    print(f"    🔵 Cosine weight: {stats['cosine_mean']:.3f} ± {stats['cosine_std']:.3f}")
                    print(f"    🟢 Covariance weight: {stats['cov_mean']:.3f} ± {stats['cov_std']:.3f}")
                    print(f"    🟡 Variance weight: {stats['var_mean']:.3f} ± {stats['var_std']:.3f}")

                    # Check if weights are learning (showing variation)
                    total_std = stats['cosine_std'] + stats['cov_std'] + stats['var_std']
                    if total_std > 0.05:
                        print(f"    ✅ Dynamic weights are learning! (total_std={total_std:.3f})")
                        weight_learning_detected = True
                    else:
                        print(f"    ⚠️  Weights may be stuck (total_std={total_std:.3f})")

                    # Check component balance
                    weights = [stats['cosine_mean'], stats['cov_mean'], stats['var_mean']]
                    dominant_component = ['Cosine', 'Covariance', 'Variance'][np.argmax(weights)]
                    print(f"    🏆 Dominant component: {dominant_component} ({max(weights):.3f})")

                module.clear_weight_history()
                module.record_weights = False

    # Overall assessment
    avg_variance = np.mean(total_score_vars)
    max_variance = max(total_score_vars) if total_score_vars else 0

    print(f"\n📊 OVERALL ASSESSMENT:")
    print(f"  Average score variance: {avg_variance:.6f}")
    print(f"  Maximum score variance: {max_variance:.6f}")
    print(f"  Dynamic weight learning: {'✅ YES' if weight_learning_detected else '❌ NO'}")

    # Determine breakthrough status
    if avg_variance > 0.01:
        print(f"  🎉 SUCCESS: Model has broken the variance collapse!")
        breakthrough_status = "SUCCESS"
    elif avg_variance > 0.001:
        print(f"  🟡 PROGRESS: Variance increasing, keep training!")
        breakthrough_status = "PROGRESS"  
    else:
        print(f"  🔴 ISSUE: Still zero variance, need more aggressive fixes")
        breakthrough_status = "STUCK"

    return avg_variance, breakthrough_status

def train(base_loader, val_loader, model, optimization, num_epoch, params):
    """TIER 3+ Enhanced training function with advanced monitoring"""
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

    # TIER 3+: Adaptive learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=params.learning_rate * 0.01)

    max_acc = 0
    max_variance = 0
    train_losses = []
    val_accuracies = []
    variance_history = []

    print("🚀 Starting TIER 3+ Advanced training...")
    print(f"   Learning rate: {params.learning_rate}")
    print(f"   Gamma: {getattr(params, 'gamma', 'default')}")
    print(f"   Lambda reg: {getattr(params, 'lambda_reg', 'default')}")
    print(f"   Dynamic weights: {getattr(params, 'dynamic_weight', False)}")

    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0
        num_batches = 0
        gradient_norms = []

        # TIER 3+: Enhanced training loop with gradient monitoring
        for i, (x, _) in enumerate(base_loader):
            optimizer.zero_grad()

            # Get loss
            acc, loss = model.set_forward_loss(x)
            epoch_loss += loss.item()
            num_batches += 1

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  NaN/Inf loss detected at epoch {epoch}, batch {i}")
                continue

            loss.backward()

            # TIER 3+: Advanced gradient monitoring
            total_norm = 0
            param_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            total_norm = total_norm ** (1. / 2)
            gradient_norms.append(total_norm)

            # TIER 3+: Adaptive gradient clipping
            if total_norm > 2.0:
                clip_value = 0.5
            elif total_norm < 0.001:
                clip_value = 2.0
                # Add gradient noise if gradients are too small
                for p in model.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * 0.0001
                        p.grad.data.add_(noise)
            else:
                clip_value = 1.0

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            # TIER 3+: More frequent monitoring
            if i % 20 == 0:
                print(f"  Batch {i}: Loss={loss:.4f}, Acc={acc:.4f}, Grad_norm={total_norm:.4f}")

            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0
        train_losses.append(avg_loss)

        # TIER 3+: Enhanced validation with advanced monitoring
        with torch.no_grad():
            model.eval()
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc = model.val_loop(val_loader, epoch, params.wandb)
            val_accuracies.append(acc)

            # TIER 3+: Advanced monitoring every 5 epochs or early epochs
            if epoch % 5 == 0 or epoch < 10:
                avg_variance, breakthrough_status = monitor_advanced_features(
                    model, val_loader, device, epoch, max_episodes=3)
                variance_history.append(avg_variance)

                # Track variance breakthrough
                if avg_variance > max_variance:
                    max_variance = avg_variance
                    print(f"🎯 New variance record: {max_variance:.6f}")

                # TIER 3+: Adaptive learning rate based on progress
                if breakthrough_status == "STUCK" and epoch > 15:
                    current_lr = optimizer.param_groups[0]['lr']
                    new_lr = min(current_lr * 1.2, 0.002)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"🚀 Increasing LR to escape collapse: {new_lr:.6f}")
                elif breakthrough_status == "SUCCESS" and epoch > 20:
                    current_lr = optimizer.param_groups[0]['lr']
                    new_lr = max(current_lr * 0.95, params.learning_rate * 0.1)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"🎯 Fine-tuning with lower LR: {new_lr:.6f}")

            # Enhanced model saving logic
            if acc > max_acc:
                print(f"📈 New best model! Acc: {acc:.4f} (prev: {max_acc:.4f})")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({
                    'epoch': epoch, 
                    'state': model.state_dict(), 
                    'acc': acc,
                    'variance': avg_variance if 'avg_variance' in locals() else 0
                }, outfile)

            # TIER 3+: Enhanced progress reporting
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epoch}: Loss={avg_loss:.4f}, Val_Acc={acc:.4f}, "
                  f"LR={current_lr:.6f}, Grad={avg_grad_norm:.4f}")

            # TIER 3+: Progress warnings
            if epoch > 10 and acc < 0.15:
                print("⚠️  Still at random accuracy - model may need architectural changes")
            if epoch > 5 and avg_grad_norm < 0.0001:
                print("⚠️  Very small gradients - may indicate vanishing gradient problem")
            if 'avg_variance' in locals() and avg_variance == 0 and epoch > 10:
                print("⚠️  Zero variance persists - consider more aggressive parameter changes")

            if (epoch % params.save_freq == 0) or (epoch == num_epoch-1):
                outfile = os.path.join(
                    params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save(
                    {'epoch': epoch, 'state': model.state_dict(), 'acc': acc}, outfile)

        # Update learning rate (unless manually overridden above)
        scheduler.step()

        # TIER 3+: Enhanced logging
        if params.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_acc': acc,
                'learning_rate': current_lr,
                'max_acc': max_acc,
                'gradient_norm': avg_grad_norm,
                'max_variance': max_variance
            })

        print()

    # TIER 3+: Final training summary
    print("📊 TIER 3+ Training Summary:")
    print(f"   Best validation accuracy: {max_acc:.4f}")
    print(f"   Maximum score variance: {max_variance:.6f}")
    print(f"   Final training loss: {train_losses[-1]:.4f}")
    print(f"   Average gradient norm: {np.mean(gradient_norms):.6f}")

    return model

def direct_test(test_loader, model, params):
    """Enhanced testing with better error handling and debugging"""
    correct = 0
    count = 0
    acc = []
    iter_num = len(test_loader)
    class_predictions = torch.zeros(params.n_way)

    print("🧪 Running direct test...")

    with tqdm.tqdm(total=len(test_loader)) as pbar:
        for i, (x, _) in enumerate(test_loader):
            # Process in smaller chunks to avoid OOM
            if x.size(0) > 16:  # If batch is larger than 16
                scores_list = []
                chunk_size = 16
                for j in range(0, x.size(0), chunk_size):
                    x_chunk = x[j:j+chunk_size].to(device)
                    with torch.no_grad():  # Ensure no gradients
                        scores_chunk = model.set_forward(x_chunk)
                        scores_list.append(scores_chunk.cpu())
                    torch.cuda.empty_cache()  # Clear cache after each chunk
                scores = torch.cat(scores_list, dim=0)
            else:
                with torch.no_grad():  # Ensure no gradients
                    x = x.to(device)
                    scores = model.set_forward(x)

            pred = scores.data.cpu().numpy().argmax(axis=1)
            y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)

            # Track class prediction distribution
            unique, counts = np.unique(pred, return_counts=True)
            for class_idx, count in zip(unique, counts):
                if class_idx < params.n_way:
                    class_predictions[class_idx] += count

            acc.append(np.mean(pred == y)*100)
            pbar.set_description(
                'Test | Acc {:.6f}'.format(np.mean(acc)))
            pbar.update(1)

    acc_all = np.asarray(acc)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)

    # Check prediction distribution
    print(f"📊 Class prediction distribution: {class_predictions.numpy()}")
    num_predicted_classes = (class_predictions > 0).sum().item()
    if num_predicted_classes < params.n_way:
        print(f"⚠️  WARNING: Only {num_predicted_classes}/{params.n_way} classes predicted!")

    return acc_mean, acc_std

@torch.no_grad()
def evaluate(loader, model, n_way, class_names=None, chunk=16, device="cuda"):
    """Enhanced evaluation function with better error handling"""
    model.eval()
    all_true, all_pred, times = [], [], []

    for x, _ in loader:                     # dataset's y is ignored
        t0 = time.time()
        try:
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
        except Exception as e:
            print(f"Error in evaluation: {e}")
            continue

    if not all_true or not all_pred:
        print("Error: No valid predictions collected!")
        return {}

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    try:
        res = dict(
            macro_f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            class_f1   = f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            conf_mat   = confusion_matrix(y_true, y_pred).tolist(),
            avg_inf_time = float(np.mean(times)) if times else 0.0,
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
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {}

    return res

def pretty_print(res):
    """Enhanced pretty printing with more information"""
    if not res:
        print("No results to display!")
        return

    print(f"\n📊 EVALUATION RESULTS:")
    print("=" * 50)
    print(f"🎯 Macro-F1: {res['macro_f1']:.4f}")
    print("\n📈 Per-class F1 scores:")
    for name, f in zip(res["class_names"], res["class_f1"]):
        print(f"   F1 '{name}': {f:.4f}")

    print("\n🔢 Confusion matrix:")
    print(np.array(res["conf_mat"]))
    print(f"\n⏱️  Avg inference time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"💾 Model size: {res['param_count']:.2f} M params")
    print(f"🖥️  GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']}/{res['gpu_mem_total_MB']} MB")
    print(f"🖥️  CPU util: {res['cpu_util']}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")
    print("=" * 50)

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
    """Enhanced seed function for better reproducibility"""
    seed = 4040
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(10)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"🌱 Random seeds set: torch={seed}, numpy=10, random={seed}")

def change_model(model_name):
    """Model name conversion for backbone compatibility"""
    if model_name == 'Conv4':
        model_name = 'Conv4NP'
    elif model_name == 'Conv6':
        model_name = 'Conv6NP'
    elif model_name == 'Conv4S':
        model_name = 'Conv4SNP'
    elif model_name == 'Conv6S':
        model_name = 'Conv6SNP'
    return model_name

def debug_model_predictions(model, test_loader, device='cuda', max_episodes=5):
    """Enhanced debugging function to analyze model predictions"""
    model.eval()
    print("\n🔍 DEBUGGING MODEL PREDICTIONS:")
    print("=" * 50)

    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= max_episodes:
                break

            x = x.to(device)
            scores = model.set_forward(x)

            # Get targets
            n_way = scores.size(1)
            n_query = scores.size(0) // n_way
            target = torch.repeat_interleave(torch.arange(n_way), n_query).to(device)

            predictions = torch.argmax(scores, dim=1)

            print(f"Episode {i+1}:")
            print(f"  📊 Scores shape: {scores.shape}")
            print(f"  📈 Score range: [{scores.min():.3f}, {scores.max():.3f}]")
            print(f"  📉 Score std: {scores.std():.3f}")
            print(f"  🎯 Predictions: {predictions.cpu().numpy()}")
            print(f"  ✅ Targets: {target.cpu().numpy()}")
            print(f"  🔢 Unique predictions: {torch.unique(predictions).cpu().numpy()}")
            print(f"  ✔️  Accuracy: {(predictions == target).float().mean():.3f}")
            print()

            # Check for problematic patterns
            if len(torch.unique(predictions)) < n_way:
                print(f"  ⚠️  WARNING: Only predicting {len(torch.unique(predictions))} out of {n_way} classes!")

            if torch.std(scores) < 0.1:
                print(f"  ⚠️  WARNING: Scores have very low variance ({scores.std():.3f})")

            if torch.any(torch.isnan(scores)) or torch.any(torch.isinf(scores)):
                print(f"  ❌ ERROR: NaN or Inf values detected in scores!")

    print("=" * 50)

def quick_accuracy_test(model, test_loader, device='cuda', n_episodes=10):
    """Quick test to verify the model is working properly"""
    model.eval()
    correct = 0
    total = 0
    class_correct = torch.zeros(5)
    class_total = torch.zeros(5)

    print("\n🧪 QUICK ACCURACY TEST:")
    print("=" * 30)

    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= n_episodes:
                break

            x = x.to(device)
            scores = model.set_forward(x)
            pred = torch.argmax(scores, dim=1)

            n_way = scores.size(1)
            n_query = scores.size(0) // n_way
            target = torch.repeat_interleave(torch.arange(n_way), n_query).to(device)

            correct += (pred == target).sum().item()
            total += target.size(0)

            # Per-class accuracy
            for j in range(min(n_way, 5)):
                mask = (target == j)
                if mask.sum() > 0:
                    class_correct[j] += (pred[mask] == target[mask]).sum().item()
                    class_total[j] += mask.sum().item()

    overall_acc = 100 * correct / total
    print(f"📈 Overall Accuracy: {overall_acc:.2f}%")
    print("📊 Per-class Accuracy:")
    for i in range(5):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"   Class {i}: {class_acc:.2f}%")
        else:
            print(f"   Class {i}: No samples")

    # Health check
    if overall_acc > 25:  # Better than random for 5-way
        print("✅ Model appears to be working!")
    elif overall_acc > 15:
        print("⚠️  Model is learning but still has issues")
    else:
        print("❌ Model is not learning - still at random chance")

    print("=" * 30)
    return overall_acc / 100

if __name__ == '__main__':
    params = parse_args()

    # TIER 3+ ADVANCED: Apply aggressive parameters with dynamic features
    print("🚀 TIER 3+ ADVANCED: AGGRESSIVE PARAMETERS + DYNAMIC FEATURES")
    print("="*70)

    # Apply TIER 3+ aggressive parameters
    original_lr = params.learning_rate
    params.learning_rate = 0.001                    # Higher learning rate
    params.weight_decay = 0.00001                   # Much lower weight decay

    # TIER 3+ Advanced parameters for complex formulas
    params.gamma = 0.01                             # Much smaller for stability
    params.lambda_reg = 0.001                       # Much smaller for stability
    params.initial_cov_weight = 0.01                # Smaller covariance weight
    params.initial_var_weight = 0.01                # Smaller variance weight
    params.dynamic_weight = True                    # ENABLE dynamic weights!

    print(f"✅ Learning rate: {original_lr} → {params.learning_rate} (AGGRESSIVE)")
    print(f"✅ Weight decay: {params.weight_decay} (MUCH LOWER)")
    print(f"✅ Gamma: {params.gamma} (MUCH SMALLER)")
    print(f"✅ Lambda reg: {params.lambda_reg} (MUCH SMALLER)")
    print(f"✅ Dynamic weights: {params.dynamic_weight} (ENABLED)")
    print("="*70)

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

        # Add TIER 3+ indicator
        wandb_name += "_TIER3PLUS_ADVANCED"

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

            # TIER 3+ ADVANCED: Apply all aggressive parameters with dynamic features
            model = FewShotTransformer(
                feature_model, 
                variant=variant, 
                gamma=params.gamma,                         # AGGRESSIVE: 0.01
                lambda_reg=params.lambda_reg,               # AGGRESSIVE: 0.001
                initial_cov_weight=params.initial_cov_weight, # AGGRESSIVE: 0.01
                initial_var_weight=params.initial_var_weight, # AGGRESSIVE: 0.01
                dynamic_weight=params.dynamic_weight,        # ENABLED: True
                **few_shot_params
            )

            print("✅ FewShotTransformer initialized with TIER 3+ ADVANCED features")
            print("🎛️  Dynamic weights: ENABLED")
            print("📊 Complex covariance/variance formulas: ENABLED")
            print("🔧 All numerical stability fixes: APPLIED")

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

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🏗️  Model created: {total_params/1e6:.2f}M total params, {trainable_params/1e6:.2f}M trainable")

        params.checkpoint_dir = '%sc/%s/%s_%s' % (
            configs.save_dir, params.dataset, params.backbone, params.method)
        if params.train_aug:
            params.checkpoint_dir += '_aug'
        if params.FETI and 'ResNet' in params.backbone:
            params.checkpoint_dir += '_FETI'

        params.checkpoint_dir += '_%dway_%dshot_TIER3PLUS' % (
            params.n_way, params.k_shot)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        print("===================================")
        print("🚀 TIER 3+ ADVANCED TRAINING PHASE:")
        print("===================================")

        # Initial model check with advanced monitoring
        print("\n🔍 Initial TIER 3+ model check:")
        quick_accuracy_test(model, val_loader, device, n_episodes=3)

        # Initial advanced monitoring
        print("\n🔍 Pre-training advanced analysis:")
        monitor_advanced_features(model, val_loader, device, 0, max_episodes=2)

        model = train(base_loader, val_loader, model, optimization, params.num_epoch, params)

        ######################################################################
        print("===================================")
        print("🧪 TEST PHASE:")
        print("===================================")

        # Clear CUDA cache to free up memory
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

        # Get class names for the test dataset
        try:
            class_names = get_class_names_from_file(testfile, params.n_way)
            print(f"📝 Using class names: {class_names}")
        except:
            # Fallback to generic names
            class_names = [f"Class_{i}" for i in range(params.n_way)]
            print(f"📝 Using generic class names: {class_names}")

        if params.save_iter != -1:
            modelfile = get_assigned_file(params.checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(params.checkpoint_dir)

        test_datamgr = SetDataManager(
            image_size, n_episode=iter_num, **few_shot_params)
        test_loader = test_datamgr.get_data_loader(testfile, aug=False)

        model = model.to(device)

        if modelfile is not None:
            print(f"📁 Loading model from: {modelfile}")
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])
            if 'acc' in tmp:
                print(f"📈 Best training accuracy: {tmp['acc']:.4f}")
            if 'variance' in tmp:
                print(f"📊 Best score variance: {tmp['variance']:.6f}")

        split = params.split
        if params.save_iter != -1:
            split_str = split + "_" + str(params.save_iter)
        else:
            split_str = split

        # TIER 3+: Pre-test advanced monitoring
        print("\n🔍 Pre-test TIER 3+ advanced check:")
        final_variance, final_status = monitor_advanced_features(model, test_loader, device, "FINAL", max_episodes=3)

        # Original accuracy test
        print("\n=== Standard Accuracy Test ===")
        acc_mean, acc_std = direct_test(test_loader, model, params)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))

        # Enhanced metric evaluation
        print("\n=== Detailed Metric Assessment ===")
        res = evaluate(test_loader, model, params.n_way, class_names=class_names, device=device)
        pretty_print(res)

        # TIER 3+: Final comprehensive assessment
        print("\n🎯 TIER 3+ ADVANCED FINAL ASSESSMENT:")
        print("="*50)
        print(f"📊 Final accuracy: {acc_mean:.2f}%")
        print(f"📈 Final variance: {final_variance:.6f}")
        print(f"🎛️  Dynamic weights status: {final_status}")

        if acc_mean > 40 and final_variance > 0.01:
            print("\n🎉 SUCCESS: TIER 3+ Advanced breakthrough achieved!")
            print("✅ High accuracy + healthy variance + dynamic features working!")
        elif acc_mean > 25:
            print("\n✅ GOOD PROGRESS: Model performing above random chance")
            if final_variance < 0.01:
                print("⚠️  But variance still low - complex formulas need more work")
        else:
            print("\n⚠️  PARTIAL SUCCESS: Still need more tuning")

        if params.wandb and res:
            wandb.log({
                'Test Acc': acc_mean,
                'Macro F1': res.get('macro_f1', 0),
                'Final Variance': final_variance,
                'Breakthrough Status': final_status,
                'Avg Inference Time (ms)': res.get('avg_inf_time', 0) * 1000,
                'Model Size (M)': res.get('param_count', 0),
                'Solution': 'TIER3_Plus_Advanced'
            })

            # Log per-class F1 scores
            if 'class_f1' in res and 'class_names' in res:
                for i, (name, f1) in enumerate(zip(res['class_names'], res['class_f1'])):
                    wandb.log({f'F1_Class_{name}': f1})

        # Save results to file
        if res:
            with open('./record/results.txt', 'a') as f:
                timestamp = params.datetime
                aug_str = '-aug' if params.train_aug else ''
                aug_str += '-FETI' if params.FETI and 'ResNet' in params.backbone else ''

                if params.backbone == "Conv4SNP":
                    params.backbone = "Conv4"
                elif params.backbone == "Conv6SNP":
                    params.backbone = "Conv6"

                exp_setting = '%s-%s-%s%s-%sw%ss-TIER3PLUS' % (params.dataset, params.backbone,
                                                     params.method, aug_str, params.n_way, params.k_shot)

                acc_str = 'Test Acc = %4.2f%% +- %4.2f%% | Macro F1 = %4.4f | Variance = %4.6f' % (
                    acc_mean, 1.96 * acc_std/np.sqrt(iter_num), res.get('macro_f1', 0), final_variance)

                f.write('Time: %s Setting: %s %s \n' % (timestamp, exp_setting.ljust(50), acc_str))

                # Write detailed per-class F1 scores
                f.write('Per-class F1 scores: ')
                for name, f1 in zip(res.get('class_names', []), res.get('class_f1', [])):
                    f.write(f'{name}={f1:.4f} ')
                f.write('\n')

        if params.wandb:
            wandb.finish()

    print("\n🎉 TIER 3+ ADVANCED TRAINING AND TESTING COMPLETED!")
    print("🎛️  Dynamic features, complex formulas, and aggressive parameters applied!")
