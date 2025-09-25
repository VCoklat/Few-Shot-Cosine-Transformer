
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

# ===============================
# PROGRESSIVE REGULARIZATION SYSTEM
# ===============================

class ProgressiveRegularizationScheduler:
    """
    Implements progressive regularization that starts with minimal regularization
    and gradually increases complexity over training epochs.
    """
    def __init__(self, 
                 total_epochs,
                 minimal_reg_epochs=20,      # First 20 epochs with minimal regularization
                 base_dropout=0.02,          # Very low initial dropout
                 target_dropout=0.12,        # Moderate final dropout
                 base_weight_decay=0.000001, # Extremely low initial weight decay
                 target_weight_decay=0.0005, # Moderate final weight decay
                 base_temperature=1.0,
                 target_temperature=0.3):

        self.total_epochs = total_epochs
        self.minimal_reg_epochs = min(minimal_reg_epochs, total_epochs // 2)
        self.gradual_epochs = max(1, total_epochs - self.minimal_reg_epochs)

        # Store parameters
        self.base_dropout = base_dropout
        self.target_dropout = target_dropout
        self.base_weight_decay = base_weight_decay
        self.target_weight_decay = target_weight_decay
        self.base_temperature = base_temperature
        self.target_temperature = target_temperature

        # Performance tracking
        self.val_performance_history = []
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.patience_threshold = 5

    def get_current_regularization(self, epoch):
        """Returns current regularization parameters based on epoch"""

        if epoch < self.minimal_reg_epochs:
            # Phase 1: Minimal regularization - let the model learn basic patterns
            return {
                'dropout_rate': self.base_dropout,
                'weight_decay': self.base_weight_decay,
                'temperature': self.base_temperature,
                'use_interventions': False,  # Disable aggressive interventions
                'phase': 'minimal'
            }

        else:
            # Phase 2: Gradual increase using smooth interpolation
            progress = min(1.0, (epoch - self.minimal_reg_epochs) / self.gradual_epochs)
            smooth_progress = 0.5 * (1 + np.cos(np.pi * (1 - progress)))  # Cosine schedule

            current_dropout = self.base_dropout + smooth_progress * (self.target_dropout - self.base_dropout)
            current_weight_decay = self.base_weight_decay + smooth_progress * (self.target_weight_decay - self.base_weight_decay)
            current_temperature = self.base_temperature + smooth_progress * (self.target_temperature - self.base_temperature)

            return {
                'dropout_rate': current_dropout,
                'weight_decay': current_weight_decay,
                'temperature': current_temperature,
                'use_interventions': progress > 0.5,  # Enable interventions only in later phases
                'phase': 'gradual',
                'progress': progress
            }

    def update_validation_performance(self, val_acc):
        """Track validation performance and provide adaptive feedback"""
        self.val_performance_history.append(val_acc)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Determine if regularization should be reduced
        should_reduce = (self.patience_counter > self.patience_threshold and 
                        len(self.val_performance_history) >= 3 and
                        np.mean(self.val_performance_history[-3:]) < np.mean(self.val_performance_history[-6:-3]))

        return {
            'best_val_acc': self.best_val_acc,
            'patience_counter': self.patience_counter,
            'should_reduce_regularization': should_reduce
        }

def update_model_regularization(model, reg_params):
    """Update model components with current regularization parameters"""

    # Update dropout rates in all dropout layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = reg_params['dropout_rate']

    # Update temperature parameters if they exist
    if hasattr(model, 'temperature_sm'):
        with torch.no_grad():
            current_temp = model.temperature_sm.item()
            target_temp = reg_params['temperature']
            # Smooth temperature adjustment
            new_temp = current_temp * 0.9 + target_temp * 0.1
            model.temperature_sm.data.fill_(new_temp)

    # Control intervention system
    if hasattr(model, 'intervention_tracker'):
        if not reg_params['use_interventions']:
            # Disable interventions during minimal phase
            model.intervention_tracker.intervention_cooldown = 100  # Long cooldown
        else:
            # Normal intervention behavior
            if model.intervention_tracker.intervention_cooldown > 10:
                model.intervention_tracker.intervention_cooldown = 0

def train_with_progressive_regularization(base_loader, val_loader, model, optimization, num_epoch, params):
    """Enhanced training with progressive regularization scheduler - FIXED tensor comparison"""

    print("🎯 PROGRESSIVE REGULARIZATION TRAINING ACTIVATED!")
    print("="*60)

    # Initialize progressive regularization scheduler
    scheduler = ProgressiveRegularizationScheduler(
        total_epochs=num_epoch,
        minimal_reg_epochs=min(20, num_epoch // 2),  # First 20 epochs or half of training
        base_dropout=0.02,           # Very minimal initially
        target_dropout=0.12,         # Moderate final dropout
        base_weight_decay=0.000001,  # Extremely low initially  
        target_weight_decay=0.0005,  # Moderate final weight decay
        base_temperature=1.0,        # Normal temperature initially
        target_temperature=0.3       # Lower final temperature
    )

    # Setup optimizer with initial minimal parameters
    initial_reg = scheduler.get_current_regularization(0)

    # FIXED: Better parameter separation using IDs instead of tensor comparison
    dynamic_param_ids = set()
    dynamic_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'weight_predictor' in name or 'weight_temperature' in name:
            dynamic_params.append(param)
            dynamic_param_ids.add(id(param))  # Use parameter ID for comparison
        else:
            other_params.append(param)

    # Initialize optimizer with progressive parameters
    if optimization == 'Adam':
        if dynamic_params:
            optimizer = torch.optim.Adam([
                {'params': other_params, 'lr': params.learning_rate, 'weight_decay': initial_reg['weight_decay']},
                {'params': dynamic_params, 'lr': params.learning_rate * 2.0, 'weight_decay': initial_reg['weight_decay'] * 0.1}
            ])
        else:
            optimizer = torch.optim.Adam(model.parameters(), 
                                       lr=params.learning_rate, 
                                       weight_decay=initial_reg['weight_decay'])
    elif optimization == 'AdamW':
        if dynamic_params:
            optimizer = torch.optim.AdamW([
                {'params': other_params, 'lr': params.learning_rate, 'weight_decay': initial_reg['weight_decay']},
                {'params': dynamic_params, 'lr': params.learning_rate * 2.0, 'weight_decay': initial_reg['weight_decay'] * 0.1}
            ])
        else:
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=params.learning_rate, 
                                        weight_decay=initial_reg['weight_decay'])
    elif optimization == 'SGD':
        if dynamic_params:
            optimizer = torch.optim.SGD([
                {'params': other_params, 'lr': params.learning_rate, 'momentum': params.momentum, 'weight_decay': initial_reg['weight_decay']},
                {'params': dynamic_params, 'lr': params.learning_rate * 1.5, 'momentum': 0.5, 'weight_decay': initial_reg['weight_decay'] * 0.1}
            ])
        else:
            optimizer = torch.optim.SGD(model.parameters(), 
                                      lr=params.learning_rate, 
                                      momentum=params.momentum, 
                                      weight_decay=initial_reg['weight_decay'])
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    # Learning rate scheduler
    lr_scheduler_obj = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=params.learning_rate * 0.01)

    # Training tracking
    max_acc = 0
    train_losses = []
    val_accuracies = []
    regularization_history = []

    print("🎯 PROGRESSIVE TRAINING PARAMETERS:")
    print(f"   Phase 1 (Epochs 0-{scheduler.minimal_reg_epochs-1}): Minimal regularization")
    print(f"   Phase 2 (Epochs {scheduler.minimal_reg_epochs}-{num_epoch-1}): Gradual increase")
    print(f"   Initial dropout: {initial_reg['dropout_rate']:.4f}")
    print(f"   Initial weight decay: {initial_reg['weight_decay']:.6f}")
    print(f"   Dynamic parameters: {len(dynamic_params)}")
    print("="*60)

    for epoch in range(num_epoch):
        # Get current regularization parameters
        reg_params = scheduler.get_current_regularization(epoch)
        regularization_history.append(reg_params.copy())

        # Update model regularization settings
        update_model_regularization(model, reg_params)

        # FIXED: Update optimizer weight decay using parameter IDs
        for group_idx, group in enumerate(optimizer.param_groups):
            # Check if this group contains dynamic parameters
            group_has_dynamic = any(id(p) in dynamic_param_ids for p in group['params'])

            if group_has_dynamic:
                group['weight_decay'] = reg_params['weight_decay'] * 0.1  # Lower for dynamic params
            else:
                group['weight_decay'] = reg_params['weight_decay']

        # Set epoch counter for model
        if hasattr(model, 'training_epoch'):
            model.training_epoch = epoch
        model.train()

        # Print phase information
        if epoch == 0 or epoch == scheduler.minimal_reg_epochs or (epoch > 0 and reg_params['phase'] != regularization_history[epoch-1]['phase']):
            print(f"\n🔄 ENTERING {reg_params['phase'].upper()} PHASE (Epoch {epoch})")
            print(f"   Dropout: {reg_params['dropout_rate']:.4f}")
            print(f"   Weight Decay: {reg_params['weight_decay']:.6f}")
            print(f"   Interventions: {'Enabled' if reg_params['use_interventions'] else 'Disabled'}")

        # Training loop
        epoch_loss = 0
        num_batches = 0
        gradient_norms = []

        for i, (x, _) in enumerate(base_loader):
            optimizer.zero_grad()

            try:
                acc, loss = model.set_forward_loss(x)

                # Skip invalid batches
                if torch.isnan(loss) or torch.isinf(loss) or loss > 20.0:
                    print(f"⚠️  Skipping invalid batch - Loss: {loss.item():.4f}")
                    continue

                epoch_loss += loss.item()
                num_batches += 1

                loss.backward()

                # Gradient monitoring
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                          5.0 if reg_params['phase'] == 'minimal' else 2.0)
                gradient_norms.append(total_norm)

                optimizer.step()

                # Progress reporting
                if i % 20 == 0:
                    print(f"   Batch {i}: Loss={loss:.4f}, Acc={acc:.4f}, GradNorm={total_norm:.4f}")

            except Exception as e:
                print(f"⚠️  Error in batch {i}: {e}")
                continue

        if num_batches == 0:
            print(f"❌ No valid batches in epoch {epoch}")
            continue

        avg_loss = epoch_loss / num_batches
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0
        train_losses.append(avg_loss)

        # Validation
        with torch.no_grad():
            model.eval()
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc = model.val_loop(val_loader, epoch, params.wandb)
            val_accuracies.append(acc)

            # Update scheduler with validation performance
            perf_update = scheduler.update_validation_performance(acc)

            # Adaptive regularization adjustment
            if perf_update['should_reduce_regularization'] and reg_params['phase'] == 'gradual':
                print(f"📉 Performance declining - reducing regularization strength")
                # Temporarily reduce regularization
                for group in optimizer.param_groups:
                    group['weight_decay'] *= 0.8

        # Model saving
        if acc > max_acc:
            print(f"📈 NEW BEST MODEL! Acc: {acc:.4f} (prev: {max_acc:.4f})")
            max_acc = acc

            # Save best model with regularization info
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({
                'epoch': epoch,
                'state': model.state_dict(),
                'acc': acc,
                'regularization_phase': reg_params['phase'],
                'regularization_params': reg_params
            }, outfile)

        # Progress reporting
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epoch}: Loss={avg_loss:.4f}, Val_Acc={acc:.4f}, "
              f"LR={current_lr:.6f}, Phase={reg_params['phase']}")

        # Detailed regularization info every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"📊 Regularization Status:")
            print(f"   Phase: {reg_params['phase']}")
            print(f"   Dropout: {reg_params['dropout_rate']:.4f}")
            print(f"   Weight Decay: {reg_params['weight_decay']:.6f}")
            print(f"   Best Val Acc: {perf_update['best_val_acc']:.4f}")
            print(f"   Patience: {perf_update['patience_counter']}/{scheduler.patience_threshold}")

        # Checkpoint saving
        if (epoch % params.save_freq == 0) or (epoch == num_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, f'{epoch:d}.tar')
            torch.save({
                'epoch': epoch, 
                'state': model.state_dict(), 
                'acc': acc,
                'regularization_history': regularization_history
            }, outfile)

        # Update learning rate (not during emergency interventions)
        if not (reg_params['phase'] == 'minimal' and perf_update.get('should_reduce_regularization', False)):
            lr_scheduler_obj.step()

        # Wandb logging
        if params.wandb:
            log_data = {
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_acc': acc,
                'learning_rate': current_lr,
                'max_acc': max_acc,
                'gradient_norm': avg_grad_norm,
                'regularization_phase': reg_params['phase'],
                'dropout_rate': reg_params['dropout_rate'],
                'weight_decay': reg_params['weight_decay'],
                'temperature': reg_params['temperature'],
                'patience_counter': perf_update['patience_counter'],
                'best_val_acc': perf_update['best_val_acc']
            }

            if reg_params['phase'] == 'gradual':
                log_data['gradual_progress'] = reg_params.get('progress', 0)

            wandb.log(log_data)

    # Final training summary
    print("\n📊 PROGRESSIVE TRAINING SUMMARY:")
    print("="*50)
    print(f"   Best validation accuracy: {max_acc:.4f}")
    print(f"   Final regularization phase: {regularization_history[-1]['phase']}")
    print(f"   Epochs in minimal phase: {scheduler.minimal_reg_epochs}")
    print(f"   Epochs in gradual phase: {max(0, num_epoch - scheduler.minimal_reg_epochs)}")

    if max_acc > 0.40:
        print("🎉 PROGRESSIVE TRAINING SUCCESS - High performance achieved!")
    elif max_acc > 0.30:
        print("✅ PROGRESSIVE TRAINING SUCCESS - Good performance achieved!")
    elif max_acc > 0.25:
        print("🔄 PROGRESSIVE TRAINING PARTIAL SUCCESS - Above random performance!")
    else:
        print("🔄 PROGRESSIVE TRAINING - Consider extending minimal phase!")

    return model

def monitor_model_health(model, test_loader, device='cuda', epoch=0, max_episodes=3):
    """Gentle monitoring function - replaces aggressive emergency monitoring"""
    model.eval()
    print(f"\n📊 Model Health Check - Epoch {epoch}")
    print("="*40)

    total_score_vars = []

    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= max_episodes:
                break

            x = x.to(device)
            scores = model.set_forward(x)

            # Gentle variance monitoring
            score_var = torch.var(scores).item()
            score_std = torch.std(scores).item()
            score_range = (scores.max() - scores.min()).item()
            total_score_vars.append(score_var)

            predictions = torch.argmax(scores, dim=1)
            unique_preds = len(torch.unique(predictions))

            print(f"Episode {i+1}:")
            print(f"  📊 Score variance: {score_var:.6f}")
            print(f"  📈 Score std: {score_std:.6f}")
            print(f"  📏 Score range: {score_range:.3f}")
            print(f"  🔢 Unique predictions: {unique_preds}")

            # Positive indicators
            if score_var > 0.01:
                print("  ✅ Good variance!")
            if unique_preds >= 4:
                print("  ✅ Good diversity!")
            if score_range > 1.0:
                print("  ✅ Good separation!")

    # Overall assessment
    avg_variance = np.mean(total_score_vars)
    print(f"\n📊 Overall Assessment:")
    print(f"  Average variance: {avg_variance:.6f}")

    if avg_variance > 0.02:
        status = "HEALTHY"
        print("  🎉 Model is learning well!")
    elif avg_variance > 0.005:
        status = "PROGRESS"
        print("  🟡 Model making progress!")
    else:
        status = "NEEDS_HELP"
        print("  ⚠️  Model needs more training time")

    return avg_variance, status

# ===============================
# TESTING AND EVALUATION FUNCTIONS
# ===============================

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
    """Quick test with gentle feedback"""
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
        print("❌ NO VALID PREDICTIONS")
        return 0.0

    overall_acc = 100 * correct / total
    print(f"📈 Overall Accuracy: {overall_acc:.2f}%")

    print("📊 Per-class Accuracy:")
    for i in range(5):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  Class {i}: {class_acc:.2f}%")
        else:
            print(f"  Class {i}: No samples")

    # Health check
    if overall_acc > 35:
        print("🎉 Excellent performance!")
    elif overall_acc > 25:
        print("✅ Good progress!")
    elif overall_acc > 18:
        print("⚠️ Above random chance")
    else:
        print("🔄 Need more training")

    print("=" * 30)
    return overall_acc / 100

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == '__main__':
    params = parse_args()

    # PROGRESSIVE: Moderate parameters instead of ultra-aggressive
    print("🎯 PROGRESSIVE REGULARIZATION PROTOCOL: GRADUAL COMPLEXITY INCREASE")
    print("="*70)

    original_lr = params.learning_rate
    params.learning_rate = 0.001  # Moderate learning rate
    params.weight_decay = 0.000001  # Start very low, will increase gradually

    # More moderate complex formula parameters
    params.gamma = 0.01  # Moderate
    params.lambda_reg = 0.0001  # Very small initially
    params.initial_cov_weight = 0.01  
    params.initial_var_weight = 0.01  
    params.dynamic_weight = True  # Keep enabled

    print(f"🎯 Learning rate: {original_lr} → {params.learning_rate} (MODERATE)")
    print(f"🎯 Weight decay: {params.weight_decay} (STARTS MINIMAL)")
    print(f"🎯 Will gradually increase regularization over {params.num_epoch} epochs")
    print("="*70)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(params))
    print()

    # Wandb setup
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
        wandb_name += "_PROGRESSIVE_REG"  # Changed from EMERGENCY

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

            # Initialize with moderate parameters
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

            print("🎯 FewShotTransformer initialized")
            print("🎯 Moderate parameters applied")
            print("🎯 Progressive regularization systems active")

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
    print(f"🎯 Model: {total_params/1e6:.2f}M total params, {trainable_params/1e6:.2f}M trainable")

    params.checkpoint_dir = '%sc/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.backbone, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if params.FETI and 'ResNet' in params.backbone:
        params.checkpoint_dir += '_FETI'
    params.checkpoint_dir += '_%dway_%dshot_PROGRESSIVE' % (
        params.n_way, params.k_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    print("=====================================")
    print("🎯 PROGRESSIVE TRAINING PROTOCOL:")
    print("=====================================")

    # Initial model check (gentler than emergency version)
    print("\n🎯 Initial model assessment:")
    initial_acc = quick_accuracy_test(model, val_loader, device, n_episodes=3)

    # PROGRESSIVE training (replace emergency training)
    model = train_with_progressive_regularization(base_loader, val_loader, model, optimization, params.num_epoch, params)

    ######################################################################
    print("=========================================")
    print("🧪 TEST PHASE:")
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
        if 'regularization_phase' in tmp:
            print(f"🎯 Final regularization phase: {tmp['regularization_phase']}")

    # Final health check
    print("\n🎯 Pre-test model health check:")
    final_variance, final_status = monitor_model_health(model, test_loader, device, "FINAL", max_episodes=3)

    print("\n=== ACCURACY TEST ===")
    acc_mean, acc_std = direct_test(test_loader, model, params)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %
          (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))

    print("\n=== DETAILED ASSESSMENT ===")
    res = evaluate(test_loader, model, params.n_way, class_names=class_names, device=device)
    pretty_print(res)

    # Final assessment
    print("\n🎯 PROGRESSIVE REGULARIZATION FINAL ASSESSMENT:")
    print("="*50)
    print(f"📊 Final accuracy: {acc_mean:.2f}%")
    print(f"📈 Final variance: {final_variance:.6f}")
    print(f"🎯 Health status: {final_status}")

    if acc_mean > 40 and final_variance > 0.01:
        print("\n🎉 PROGRESSIVE REGULARIZATION COMPLETE SUCCESS!")
        print("✅ High accuracy + healthy variance achieved!")
        success_status = "COMPLETE_SUCCESS"
    elif acc_mean > 30:
        print("\n🎉 PROGRESSIVE REGULARIZATION SUCCESS!")
        print("✅ Good performance achieved!")
        success_status = "SUCCESS"
    elif acc_mean > 25:
        print("\n✅ PROGRESSIVE REGULARIZATION PARTIAL SUCCESS!")
        print("⚠️ Above random chance - good foundation!")
        success_status = "PARTIAL_SUCCESS"
    else:
        print("\n🔄 PROGRESSIVE REGULARIZATION IN PROGRESS!")
        print("❓ Consider extending training or adjusting parameters!")
        success_status = "NEEDS_MORE_TRAINING"

    # Logging and saving
    if params.wandb and res:
        wandb.log({
            'Progressive_Final_Acc': acc_mean,
            'Progressive_Variance': final_variance,
            'Health_Status': final_status,
            'Success_Status': success_status,
            'Macro_F1': res.get('macro_f1', 0),
            'Solution': 'PROGRESSIVE_REGULARIZATION'
        })

    # Save results
    if res:
        with open('./record/progressive_results.txt', 'a') as f:
            timestamp = params.datetime
            exp_setting = f'{params.dataset}-{params.backbone}-{params.method}-PROGRESSIVE-{params.n_way}w{params.k_shot}s'
            acc_str = f'Progressive Acc = {acc_mean:.2f}% +- {1.96 * acc_std/np.sqrt(iter_num):.2f}% | Status = {success_status} | Variance = {final_variance:.6f}'
            f.write(f'Time: {timestamp} Setting: {exp_setting.ljust(60)} {acc_str}\n')

    if params.wandb:
        wandb.finish()

    print(f"\n🎯 PROGRESSIVE REGULARIZATION PROTOCOL COMPLETED!")
    print(f"🎯 Status: {success_status}")
    print("="*50)
