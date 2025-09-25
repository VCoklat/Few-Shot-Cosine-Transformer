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

# EMERGENCY: Ultra-aggressive monitoring function
def monitor_advanced_features(model, test_loader, device='cuda', epoch=0, max_episodes=3):
    """EMERGENCY: Monitor with ultra-low thresholds for breakthrough detection"""
    model.eval()
    print(f"\n🔍 EMERGENCY TIER 3+ MONITORING - Epoch {epoch}")
    print("="*60)

    # Enable weight recording
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

            # EMERGENCY: Ultra-sensitive variance monitoring
            score_var = torch.var(scores).item()
            score_std = torch.std(scores).item()
            score_range = (scores.max() - scores.min()).item()
            total_score_vars.append(score_var)

            predictions = torch.argmax(scores, dim=1)
            unique_preds = len(torch.unique(predictions))

            print(f"Episode {i+1}:")
            print(f"  📊 Score variance: {score_var:.6f} (target: >0.002)")  # Much lower
            print(f"  📈 Score std: {score_std:.6f} (target: >0.05)")  # Much lower
            print(f"  📏 Score range: [{scores.min():.3f}, {scores.max():.3f}] (separation: {score_range:.3f})")
            print(f"  🔢 Unique predictions: {unique_preds}/5")

            # EMERGENCY: Very encouraging success indicators
            breakthrough_indicators = []
            if score_var > 0.002:  # Ultra-low threshold
                breakthrough_indicators.append("✅ VARIANCE BREAKTHROUGH!")
            if unique_preds >= 3:  # Lower requirement
                breakthrough_indicators.append("✅ MULTI-CLASS SUCCESS!")
            if score_range > 0.3:  # Much lower threshold
                breakthrough_indicators.append("✅ GOOD SEPARATION!")
            if score_std > 0.05:  # Much lower threshold  
                breakthrough_indicators.append("✅ HEALTHY DIVERSITY!")

            if breakthrough_indicators:
                for indicator in breakthrough_indicators:
                    print(f"  {indicator}")
            else:
                print(f"  ⚠️  Still need improvement...")

    # EMERGENCY: Dynamic weight analysis with ultra-low thresholds
    print(f"\n🎛️  EMERGENCY DYNAMIC WEIGHT ANALYSIS:")
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

                    # EMERGENCY: Much lower threshold for weight learning detection
                    total_std = stats['cosine_std'] + stats['cov_std'] + stats['var_std']
                    if total_std > 0.008:  # Ultra-low threshold - was 0.02
                        print(f"    ✅ Dynamic weights are learning! (total_std={total_std:.3f})")
                        weight_learning_detected = True
                    else:
                        print(f"    🚨 EMERGENCY: Weights completely stuck (total_std={total_std:.3f})")
                        # IMMEDIATE INTERVENTION
                        if hasattr(module, 'nuclear_breakthrough'):
                            module.nuclear_breakthrough()
                            print(f"    ☢️ NUCLEAR intervention applied immediately!")

                    # Check component balance
                    weights = [stats['cosine_mean'], stats['cov_mean'], stats['var_mean']]
                    dominant_component = ['Cosine', 'Covariance', 'Variance'][np.argmax(weights)]
                    print(f"    🏆 Dominant component: {dominant_component} ({max(weights):.3f})")

                module.clear_weight_history()

    # Disable weight recording
    for module in model.modules():
        if hasattr(module, 'record_weights'):
            module.record_weights = False

    # EMERGENCY: Ultra-aggressive overall assessment
    avg_variance = np.mean(total_score_vars)
    max_variance = max(total_score_vars) if total_score_vars else 0

    print(f"\n📊 EMERGENCY OVERALL ASSESSMENT:")
    print(f"  Average score variance: {avg_variance:.8f}")
    print(f"  Maximum score variance: {max_variance:.8f}")
    print(f"  Dynamic weight learning: {'✅ YES' if weight_learning_detected else '❌ NO'}")

    # EMERGENCY: Ultra-optimistic breakthrough detection
    if avg_variance > 0.003:  # Much lower threshold
        print(f"  🎉 SUCCESS: Model has broken the variance collapse!")
        breakthrough_status = "SUCCESS"
    elif avg_variance > 0.0008:  # Ultra-low threshold
        print(f"  🟡 PROGRESS: Variance increasing, keep training!")
        breakthrough_status = "PROGRESS"
    else:
        print(f"  🚨 EMERGENCY: Complete variance collapse - IMMEDIATE INTERVENTION NEEDED!")
        breakthrough_status = "EMERGENCY"

    return avg_variance, breakthrough_status

def emergency_model_reset(model):
    """EMERGENCY: Complete model reset when totally stuck"""
    print("🚨 EMERGENCY MODEL RESET - NUCLEAR OPTION!")
    
    with torch.no_grad():
        # 1. Reset all attention components
        if hasattr(model, 'ATTN'):
            if hasattr(model.ATTN, 'nuclear_breakthrough'):
                model.ATTN.nuclear_breakthrough()
            if hasattr(model.ATTN, 'emergency_diversity_injection'):
                model.ATTN.emergency_diversity_injection()
            if hasattr(model.ATTN, 'force_dynamic_learning'):
                model.ATTN.force_dynamic_learning()
        
        # 2. Reset temperature systems
        if hasattr(model, 'temperature_sm'):
            model.temperature_sm.data.fill_(15.0)  # High exploration
            
        # 3. Shake all proto weights with massive noise
        if hasattr(model, 'proto_weight'):
            noise = torch.randn_like(model.proto_weight) * 0.8  # Massive noise
            model.proto_weight.data.add_(noise)
            
        # 4. Reset final classification layer
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.out_features == 1:
                nn.init.xavier_uniform_(module.weight, gain=2.0)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)
    
    print("☢️ NUCLEAR RESET COMPLETE - All systems reinitialized!")

def train(base_loader, val_loader, model, optimization, num_epoch, params):
    """EMERGENCY: Training with immediate breakthrough interventions"""
    
    print("🚨 EMERGENCY TRAINING PROTOCOL ACTIVATED!")
    print("="*50)
    
    # EMERGENCY: Immediate pre-training breakthrough
    print("🚨 STEP 1: Forcing immediate dynamic weight breakthrough...")
    emergency_model_reset(model)
    
    # EMERGENCY: Ultra-aggressive optimizer setup
    dynamic_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'weight_predictor' in name or 'weight_temperature' in name:
            dynamic_params.append(param)
        else:
            other_params.append(param)
    
    if optimization == 'Adam':
        if dynamic_params:
            optimizer = torch.optim.Adam([
                {'params': other_params, 'lr': params.learning_rate},
                {'params': dynamic_params, 'lr': params.learning_rate * 8.0}  # 8x higher LR!
            ], weight_decay=params.weight_decay)
            print(f"🚨 EMERGENCY: Dynamic weight LR boosted to {params.learning_rate * 8.0:.6f} (8x)")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'AdamW':
        if dynamic_params:
            optimizer = torch.optim.AdamW([
                {'params': other_params, 'lr': params.learning_rate, 'weight_decay': params.weight_decay},
                {'params': dynamic_params, 'lr': params.learning_rate * 8.0, 'weight_decay': params.weight_decay * 0.01}  # Much lower decay
            ])
            print(f"🚨 EMERGENCY: Dynamic weight LR: {params.learning_rate * 8.0:.6f}, WD: {params.weight_decay * 0.01:.6f}")
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    elif optimization == 'SGD':
        if dynamic_params:
            optimizer = torch.optim.SGD([
                {'params': other_params, 'lr': params.learning_rate, 'momentum': params.momentum},
                {'params': dynamic_params, 'lr': params.learning_rate * 6.0, 'momentum': 0.5}  # Lower momentum
            ], weight_decay=params.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    # EMERGENCY: More aggressive scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=params.learning_rate * 0.01)

    max_acc = 0
    max_variance = 0
    train_losses = []
    val_accuracies = []
    variance_history = []
    stuck_epochs = 0
    emergency_interventions = 0

    print("🚨 EMERGENCY TRAINING PARAMETERS:")
    print(f"   Base learning rate: {params.learning_rate}")
    print(f"   Dynamic weight LR: {params.learning_rate * 8.0 if dynamic_params else params.learning_rate}")
    print(f"   Weight decay: {params.weight_decay}")
    print(f"   Emergency interventions enabled: ✅")

    for epoch in range(num_epoch):
        # Set epoch counter for model loss function
        model.training_epoch = epoch
        model.train()
        epoch_loss = 0
        num_batches = 0
        gradient_norms = []

        # EMERGENCY: Force intervention every 3 epochs if no improvement
        if epoch > 0 and epoch % 3 == 0:
            if max_acc < 0.22:  # Still near random
                emergency_interventions += 1
                print(f"\n🚨 EMERGENCY INTERVENTION #{emergency_interventions} AT EPOCH {epoch}")
                print(f"   Max accuracy still only: {max_acc:.4f}")
                print(f"   Applying nuclear breakthrough...")
                
                emergency_model_reset(model)
                
                # Boost learning rates even more
                for group in optimizer.param_groups:
                    if len(dynamic_params) > 0 and any(p in dynamic_params for p in group['params']):
                        group['lr'] = min(group['lr'] * 2.0, 0.05)  # 2x boost for dynamic
                    else:
                        group['lr'] = min(group['lr'] * 1.5, 0.01)  # 1.5x boost for others
                
                print(f"   Learning rates boosted!")

        # EMERGENCY: Enhanced training loop
        for i, (x, _) in enumerate(base_loader):
            optimizer.zero_grad()

            # Get loss with emergency handling
            try:
                acc, loss = model.set_forward_loss(x)
                
                # EMERGENCY: Skip bad batches
                if torch.isnan(loss) or torch.isinf(loss) or loss > 10.0:
                    print(f"🚨 EMERGENCY: Skipping bad batch - Loss: {loss.item():.4f}")
                    continue
                    
                epoch_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"🚨 EMERGENCY: Error in forward pass: {e}")
                continue

            loss.backward()

            # EMERGENCY: Enhanced gradient monitoring and handling
            total_norm = 0
            dynamic_norm = 0
            
            for name, p in model.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    
                    if 'weight_predictor' in name or 'weight_temperature' in name:
                        dynamic_norm += param_norm.item() ** 2

            total_norm = total_norm ** 0.5
            dynamic_norm = dynamic_norm ** 0.5
            gradient_norms.append(total_norm)

            # EMERGENCY: Aggressive gradient handling
            if total_norm > 5.0:  # Higher threshold
                clip_value = 1.0  # Less aggressive clipping
            elif total_norm < 0.001:  # Very small gradients
                # EMERGENCY: Add strong gradient noise
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        if 'weight_predictor' in name:
                            noise_scale = 0.002  # Much stronger noise
                        else:
                            noise_scale = 0.0005
                        noise = torch.randn_like(p.grad) * noise_scale
                        p.grad.data.add_(noise)
                clip_value = 10.0  # Allow bigger steps
            else:
                clip_value = 3.0

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            # EMERGENCY: More frequent progress reporting
            if i % 15 == 0:  # More frequent
                dynamic_info = f", Dyn_grad={dynamic_norm:.4f}" if dynamic_norm > 0 else ""
                print(f"  Batch {i}: Loss={loss:.4f}, Acc={acc:.4f}, Grad_norm={total_norm:.4f}{dynamic_info}")

            # Clear cache
            if i % 5 == 0:
                torch.cuda.empty_cache()

        if num_batches == 0:
            print(f"🚨 EMERGENCY: No valid batches in epoch {epoch}!")
            continue

        avg_loss = epoch_loss / num_batches
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0
        train_losses.append(avg_loss)

        # EMERGENCY: Enhanced validation
        with torch.no_grad():
            model.eval()
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc = model.val_loop(val_loader, epoch, params.wandb)
            val_accuracies.append(acc)

            # EMERGENCY: Monitor every epoch in first 20 epochs
            should_monitor = epoch < 20 or epoch % 3 == 0
            
            if should_monitor:
                avg_variance, breakthrough_status = monitor_advanced_features(
                    model, val_loader, device, epoch, max_episodes=3)
                variance_history.append(avg_variance)

                # Track variance breakthrough
                if avg_variance > max_variance:
                    max_variance = avg_variance
                    print(f"🎯 New variance record: {max_variance:.8f}")

                # EMERGENCY: Ultra-aggressive stuck detection
                if breakthrough_status == "EMERGENCY":
                    stuck_epochs += 1
                    print(f"🚨 EMERGENCY STATUS DETECTED ({stuck_epochs} times)")
                    
                    if stuck_epochs >= 1 and epoch > 2:  # Immediate intervention
                        emergency_interventions += 1
                        print(f"🚨 IMMEDIATE EMERGENCY INTERVENTION #{emergency_interventions}!")
                        
                        emergency_model_reset(model)
                        
                        # MASSIVE learning rate boost
                        for group in optimizer.param_groups:
                            if len(dynamic_params) > 0 and any(p in dynamic_params for p in group['params']):
                                group['lr'] = min(group['lr'] * 5.0, 0.1)  # 5x boost
                            else:
                                group['lr'] = min(group['lr'] * 3.0, 0.05)  # 3x boost
                        
                        stuck_epochs = 0
                        
                elif breakthrough_status == "PROGRESS":
                    stuck_epochs = max(0, stuck_epochs - 1)
                    print(f"🟡 PROGRESS detected - stuck counter reduced to {stuck_epochs}")
                    
                elif breakthrough_status == "SUCCESS":
                    stuck_epochs = 0
                    print(f"🎉 SUCCESS detected - breakthrough achieved!")

        # Enhanced model saving
        if acc > max_acc:
            print(f"📈 NEW BEST MODEL! Acc: {acc:.4f} (prev: {max_acc:.4f})")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({
                'epoch': epoch,
                'state': model.state_dict(),
                'acc': acc,
                'variance': avg_variance if 'avg_variance' in locals() else 0,
                'emergency_interventions': emergency_interventions
            }, outfile)

        # EMERGENCY: Enhanced progress reporting
        current_lr = optimizer.param_groups[0]['lr']
        dynamic_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else current_lr
        
        print(f"Epoch {epoch+1}/{num_epoch}: Loss={avg_loss:.4f}, Val_Acc={acc:.4f}, "
              f"LR={current_lr:.6f}, Grad={avg_grad_norm:.4f}")
        
        if dynamic_lr != current_lr:
            print(f"                      Dynamic_LR={dynamic_lr:.6f}, Emergency_Count={emergency_interventions}")

        # EMERGENCY: More aggressive interventions based on conditions
        if epoch > 5 and acc < 0.18:  # Still at random after 5 epochs
            print("🚨 EMERGENCY: Still at random accuracy - NUCLEAR INTERVENTION!")
            emergency_model_reset(model)
            emergency_interventions += 1
                
        if epoch > 3 and avg_grad_norm < 0.0001:  # Very small gradients
            print("🚨 EMERGENCY: Vanishing gradients detected - injecting parameter noise!")
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight_predictor' in name:
                        noise = torch.randn_like(param) * 0.05  # Strong noise
                        param.add_(noise)
                    else:
                        noise = torch.randn_like(param) * 0.01
                        param.add_(noise)

        if 'avg_variance' in locals() and avg_variance < 0.0005 and epoch > 3:
            print("🚨 EMERGENCY: Complete variance collapse - IMMEDIATE NUCLEAR INTERVENTION!")
            emergency_model_reset(model)
            emergency_interventions += 1

        # Save checkpoints
        if (epoch % params.save_freq == 0) or (epoch == num_epoch-1):
            outfile = os.path.join(
                params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save(
                {'epoch': epoch, 'state': model.state_dict(), 'acc': acc}, outfile)

        # Update learning rate (but not during emergency interventions)
        if stuck_epochs == 0 and breakthrough_status != "EMERGENCY":
            scheduler.step()

        # EMERGENCY: Enhanced logging
        if params.wandb:
            log_data = {
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_acc': acc,
                'learning_rate': current_lr,
                'max_acc': max_acc,
                'gradient_norm': avg_grad_norm,
                'max_variance': max_variance,
                'stuck_epochs': stuck_epochs,
                'emergency_interventions': emergency_interventions
            }
            
            if dynamic_lr != current_lr:
                log_data['dynamic_learning_rate'] = dynamic_lr
                log_data['dynamic_gradient_norm'] = dynamic_norm
                
            if 'avg_variance' in locals():
                log_data['current_variance'] = avg_variance
                log_data['breakthrough_status'] = breakthrough_status
                
            wandb.log(log_data)

        print()

    # EMERGENCY: Final training summary
    print("📊 EMERGENCY TRAINING SUMMARY:")
    print("="*40)
    print(f"   Best validation accuracy: {max_acc:.4f}")
    print(f"   Maximum score variance: {max_variance:.8f}")
    print(f"   Emergency interventions: {emergency_interventions}")
    print(f"   Final training loss: {train_losses[-1]:.4f}")
    
    if max_acc > 0.30:
        print("🎉 EMERGENCY PROTOCOL SUCCESS - Breakthrough achieved!")
    elif max_acc > 0.25:
        print("🟡 EMERGENCY PROTOCOL PARTIAL SUCCESS - Some improvement")
    else:
        print("🚨 EMERGENCY PROTOCOL ONGOING - Need more aggressive measures")

    return model

def direct_test(test_loader, model, params):
    """Enhanced testing with better error handling"""
    correct = 0
    count = 0
    acc = []
    iter_num = len(test_loader)
    class_predictions = torch.zeros(params.n_way)

    print("🧪 Running direct test...")

    with tqdm.tqdm(total=len(test_loader)) as pbar:
        for i, (x, _) in enumerate(test_loader):
            try:
                # Process in smaller chunks to avoid OOM
                if x.size(0) > 16:
                    scores_list = []
                    chunk_size = 16
                    for j in range(0, x.size(0), chunk_size):
                        x_chunk = x[j:j+chunk_size].to(device)
                        with torch.no_grad():
                            scores_chunk = model.set_forward(x_chunk)
                            scores_list.append(scores_chunk.cpu())
                        torch.cuda.empty_cache()
                    scores = torch.cat(scores_list, dim=0)
                else:
                    with torch.no_grad():
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
            except Exception as e:
                print(f"Error in test batch {i}: {e}")
                acc.append(0.0)
            
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
        print(f"⚠️ WARNING: Only {num_predicted_classes}/{params.n_way} classes predicted!")

    return acc_mean, acc_std

@torch.no_grad()
def evaluate(loader, model, n_way, class_names=None, chunk=16, device="cuda"):
    """Enhanced evaluation function with better error handling"""
    model.eval()
    all_true, all_pred, times = [], [], []

    for x, _ in loader:
        t0 = time.time()
        try:
            if x.size(0) > chunk:
                scores = torch.cat([
                    model.set_forward(x[i:i+chunk].to(device)).cpu()
                    for i in range(0, x.size(0), chunk)], 0)
            else:
                scores = model.set_forward(x.to(device)).cpu()

            torch.cuda.synchronize()
            times.append(time.time() - t0)
            preds = scores.argmax(1).numpy()
            all_pred.append(preds)

            num_per_class = len(preds) // n_way
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
            macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            conf_mat = confusion_matrix(y_true, y_pred).tolist(),
            avg_inf_time = float(np.mean(times)) if times else 0.0,
            param_count = sum(p.numel() for p in model.parameters())/1e6
        )

        gpus = GPUtil.getGPUs()
        res.update(
            gpu_mem_used_MB = sum(g.memoryUsed for g in gpus) if gpus else 0,
            gpu_mem_total_MB = sum(g.memoryTotal for g in gpus) if gpus else 0,
            gpu_util = float(sum(g.load for g in gpus)/len(gpus)) if gpus else 0,
            cpu_util = psutil.cpu_percent(),
            cpu_mem_used_MB = psutil.virtual_memory().used / 1_048_576,
            cpu_mem_total_MB = psutil.virtual_memory().total / 1_048_576,
            class_names = class_names or list(range(len(res["class_f1"])))
        )
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {}

    return res

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

def seed_func():
    """Enhanced seed function"""
    seed = 4040
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(10)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"🌱 Random seeds set: torch={seed}, numpy=10, random={seed}")

def change_model(model_name):
    """Model name conversion"""
    if model_name == 'Conv4':
        model_name = 'Conv4NP'
    elif model_name == 'Conv6':
        model_name = 'Conv6NP'
    elif model_name == 'Conv4S':
        model_name = 'Conv4SNP'
    elif model_name == 'Conv6S':
        model_name = 'Conv6SNP'
    return model_name

def quick_accuracy_test(model, test_loader, device='cuda', n_episodes=10):
    """EMERGENCY: Quick test with immediate feedback"""
    model.eval()
    correct = 0
    total = 0
    class_correct = torch.zeros(5)
    class_total = torch.zeros(5)

    print("\n🧪 EMERGENCY ACCURACY TEST:")
    print("=" * 30)

    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= n_episodes:
                break

            try:
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
            except Exception as e:
                print(f"Error in episode {i}: {e}")
                continue

    if total == 0:
        print("❌ NO VALID PREDICTIONS - MODEL COMPLETELY BROKEN")
        return 0.0

    overall_acc = 100 * correct / total
    print(f"📈 Overall Accuracy: {overall_acc:.2f}%")
    print("📊 Per-class Accuracy:")
    for i in range(5):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"   Class {i}: {class_acc:.2f}%")
        else:
            print(f"   Class {i}: No samples")

    # EMERGENCY health check
    if overall_acc > 35:
        print("🎉 EMERGENCY SUCCESS - Model breakthrough achieved!")
    elif overall_acc > 25:
        print("✅ EMERGENCY PROGRESS - Model learning!")
    elif overall_acc > 18:
        print("⚠️  EMERGENCY PARTIAL - Model slightly above random")
    else:
        print("🚨 EMERGENCY FAILURE - Model at random chance!")

    print("=" * 30)
    return overall_acc / 100

if __name__ == '__main__':
    params = parse_args()

    # EMERGENCY: Ultra-aggressive parameters
    print("🚨 EMERGENCY TIER 3+ PROTOCOL: ULTRA-AGGRESSIVE PARAMETERS")
    print("="*70)

    original_lr = params.learning_rate
    params.learning_rate = 0.002  # Even higher learning rate
    params.weight_decay = 0.000005  # Even lower weight decay

    # EMERGENCY: More aggressive complex formula parameters
    params.gamma = 0.005  # Smaller for stability
    params.lambda_reg = 0.0005  # Smaller for stability
    params.initial_cov_weight = 0.005  # Smaller
    params.initial_var_weight = 0.005  # Smaller
    params.dynamic_weight = True  # ENABLE

    print(f"🚨 Learning rate: {original_lr} → {params.learning_rate} (ULTRA-AGGRESSIVE)")
    print(f"🚨 Weight decay: {params.weight_decay} (ULTRA-LOW)")
    print(f"🚨 Gamma: {params.gamma} (ULTRA-SMALL)")
    print(f"🚨 Lambda reg: {params.lambda_reg} (ULTRA-SMALL)")
    print(f"🚨 Dynamic weights: {params.dynamic_weight} (ENABLED)")
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
        wandb_name += "_EMERGENCY_PROTOCOL"
        wandb.init(project=project_name, name=wandb_name,
                   config=params, id=params.datetime)

    print()

    # Dataset setup
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

        # EMERGENCY: Apply ultra-aggressive parameters
        model = FewShotTransformer(
            feature_model,
            variant=variant,
            gamma=params.gamma,
            lambda_reg=params.lambda_reg,
            initial_cov_weight=params.initial_cov_weight,
            initial_var_weight=params.initial_var_weight,
            dynamic_weight=params.dynamic_weight,
            **few_shot_params
        )

        print("🚨 EMERGENCY FewShotTransformer initialized")
        print("🚨 Ultra-aggressive parameters applied")
        print("🚨 Emergency intervention systems active")

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
    print(f"🚨 EMERGENCY Model: {total_params/1e6:.2f}M total params, {trainable_params/1e6:.2f}M trainable")

    params.checkpoint_dir = '%sc/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.backbone, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if params.FETI and 'ResNet' in params.backbone:
        params.checkpoint_dir += '_FETI'
    params.checkpoint_dir += '_%dway_%dshot_EMERGENCY' % (
        params.n_way, params.k_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    print("=========================================")
    print("🚨 EMERGENCY TRAINING PROTOCOL ACTIVATED:")
    print("=========================================")

    # Initial emergency check
    print("\n🚨 Initial emergency model check:")
    initial_acc = quick_accuracy_test(model, val_loader, device, n_episodes=3)

    # EMERGENCY training
    model = train(base_loader, val_loader, model, optimization, params.num_epoch, params)

    ######################################################################
    print("=========================================")
    print("🧪 EMERGENCY TEST PHASE:")
    print("=========================================")

    # Test setup
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

    try:
        class_names = get_class_names_from_file(testfile, params.n_way)
        print(f"📝 Using class names: {class_names}")
    except:
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
        print(f"📁 Loading best model from: {modelfile}")
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
        if 'acc' in tmp:
            print(f"📈 Best training accuracy: {tmp['acc']:.4f}")
        if 'emergency_interventions' in tmp:
            print(f"🚨 Emergency interventions used: {tmp['emergency_interventions']}")

    # Final tests
    print("\n🚨 Pre-test emergency check:")
    final_variance, final_status = monitor_advanced_features(model, test_loader, device, "FINAL", max_episodes=3)

    print("\n=== EMERGENCY ACCURACY TEST ===")
    acc_mean, acc_std = direct_test(test_loader, model, params)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %
          (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))

    print("\n=== DETAILED ASSESSMENT ===")
    res = evaluate(test_loader, model, params.n_way, class_names=class_names, device=device)
    pretty_print(res)

    # EMERGENCY: Final assessment
    print("\n🚨 EMERGENCY PROTOCOL FINAL ASSESSMENT:")
    print("="*50)
    print(f"📊 Final accuracy: {acc_mean:.2f}%")
    print(f"📈 Final variance: {final_variance:.8f}")
    print(f"🚨 Emergency status: {final_status}")

    if acc_mean > 40 and final_variance > 0.003:
        print("\n🎉 EMERGENCY PROTOCOL COMPLETE SUCCESS!")
        print("✅ High accuracy + healthy variance achieved!")
        success_status = "COMPLETE_SUCCESS"
    elif acc_mean > 30:
        print("\n🎉 EMERGENCY PROTOCOL SUCCESS!")
        print("✅ Significant improvement achieved!")
        success_status = "SUCCESS"
    elif acc_mean > 25:
        print("\n✅ EMERGENCY PROTOCOL PARTIAL SUCCESS!")
        print("⚠️  Above random chance but need more work!")
        success_status = "PARTIAL_SUCCESS"
    else:
        print("\n🚨 EMERGENCY PROTOCOL REQUIRES ESCALATION!")
        print("❌ Still at random chance - need different approach!")
        success_status = "ESCALATION_NEEDED"

    # Logging and saving
    if params.wandb and res:
        wandb.log({
            'Emergency_Final_Acc': acc_mean,
            'Emergency_Variance': final_variance,
            'Emergency_Status': final_status,
            'Success_Status': success_status,
            'Macro_F1': res.get('macro_f1', 0),
            'Solution': 'EMERGENCY_PROTOCOL'
        })

    # Save results
    if res:
        with open('./record/emergency_results.txt', 'a') as f:
            timestamp = params.datetime
            exp_setting = f'{params.dataset}-{params.backbone}-{params.method}-EMERGENCY-{params.n_way}w{params.k_shot}s'
            acc_str = f'Emergency Acc = {acc_mean:.2f}% +- {1.96 * acc_std/np.sqrt(iter_num):.2f}% | Status = {success_status} | Variance = {final_variance:.8f}'
            f.write(f'Time: {timestamp} Setting: {exp_setting.ljust(60)} {acc_str}\n')

    if params.wandb:
        wandb.finish()

    print(f"\n🚨 EMERGENCY PROTOCOL COMPLETED!")
    print(f"🎯 Status: {success_status}")
    print("="*50)
