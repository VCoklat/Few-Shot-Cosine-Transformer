"""
Training Script for Enhanced Optimal Few-Shot Model

This script implements training with invariance learning, gradient clipping,
and cosine annealing scheduler for the enhanced few-shot model.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
import tqdm

# Import project modules
import backbone
import configs
from data.datamgr import SetDataManager
from io_utils import model_dict, parse_args
from models.optimal_fewshot_enhanced import EnhancedOptimalFewShot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, train_loader, optimizer, epoch, params, grad_clip_value=1.0):
    """
    Train for one epoch with gradient clipping.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training episodes
        optimizer: Optimizer
        epoch: Current epoch number
        params: Training parameters
        grad_clip_value: Maximum gradient norm for clipping
    
    Returns:
        avg_loss: Average loss for the epoch
        avg_acc: Average accuracy for the epoch
    """
    model.train()
    
    avg_loss = 0.0
    avg_acc = 0.0
    num_episodes = len(train_loader)
    
    # Progress bar
    pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for i, (x, _) in enumerate(pbar):
        # Move data to device
        x = x.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        acc, loss = model.set_forward_loss(x)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        # Optimizer step
        optimizer.step()
        
        # Update statistics
        avg_loss += loss.item()
        avg_acc += acc
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}'
        })
    
    avg_loss /= num_episodes
    avg_acc /= num_episodes
    
    return avg_loss, avg_acc


def evaluate(model, val_loader, params):
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: DataLoader for validation episodes
        params: Evaluation parameters
    
    Returns:
        avg_acc: Average accuracy
        confidence_interval: 95% confidence interval
    """
    model.eval()
    
    acc_all = []
    
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm.tqdm(val_loader, desc='Evaluating')):
            x = x.to(device)
            correct, total = model.correct(x)
            acc = correct / total
            acc_all.append(acc)
    
    acc_all = np.array(acc_all)
    avg_acc = np.mean(acc_all)
    std_acc = np.std(acc_all)
    
    # 95% confidence interval
    confidence_interval = 1.96 * std_acc / np.sqrt(len(acc_all))
    
    return avg_acc, confidence_interval


def train_enhanced_model(params):
    """
    Main training function for enhanced model.
    
    Args:
        params: Training parameters from argument parser
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Data loaders
    print(f"Loading {params.dataset} dataset...")
    
    # Determine data files
    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['Omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'
    
    # Determine image size
    if params.dataset == "CIFAR":
        image_size = 112 if 'ResNet' in params.backbone else 64
    else:
        image_size = 224 if 'ResNet' in params.backbone else 84
    
    base_datamgr = SetDataManager(
        image_size,
        n_way=params.n_way,
        k_shot=params.k_shot,
        n_query=params.n_query,
        n_episode=params.n_episode
    )
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    
    val_datamgr = SetDataManager(
        image_size,
        n_way=params.n_way,
        k_shot=params.k_shot,
        n_query=params.n_query,
        n_episode=params.test_iter
    )
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    
    # Model
    print(f"Creating enhanced model for {params.dataset}...")
    
    if params.backbone in model_dict:
        model_func = model_dict[params.backbone]
    else:
        print(f"Warning: Unknown backbone {params.backbone}, using Conv4")
        model_func = backbone.Conv4
    
    # Dataset-specific configurations
    dataset_configs = {
        'Omniglot': {
            'use_task_invariance': True,
            'use_multi_scale': False,
            'use_feature_augmentation': True,
            'use_prototype_refinement': False,
            'domain': 'general',
            'dropout': 0.05
        },
        'miniImagenet': {
            'use_task_invariance': True,
            'use_multi_scale': True,
            'use_feature_augmentation': True,
            'use_prototype_refinement': True,
            'domain': 'general',
            'dropout': 0.1
        },
        'HAM10000': {
            'use_task_invariance': True,
            'use_multi_scale': True,
            'use_feature_augmentation': True,
            'use_prototype_refinement': True,
            'domain': 'medical',
            'dropout': 0.2
        },
        'CUB': {
            'use_task_invariance': True,
            'use_multi_scale': True,
            'use_feature_augmentation': True,
            'use_prototype_refinement': True,
            'domain': 'fine_grained',
            'dropout': 0.15
        }
    }
    
    # Get configuration for dataset or use default
    config = dataset_configs.get(params.dataset, dataset_configs['miniImagenet'])
    
    # Override dropout if specified
    if params.dropout != 0.1:
        config['dropout'] = params.dropout
    
    # Create enhanced model with explicit configuration
    model = EnhancedOptimalFewShot(
        model_func=model_func,
        n_way=params.n_way,
        k_shot=params.k_shot,
        n_query=params.n_query,
        feature_dim=params.feature_dim,
        n_heads=params.n_heads,
        dropout=config['dropout'],
        dataset=params.dataset,
        use_task_invariance=config['use_task_invariance'],
        use_multi_scale=config['use_multi_scale'],
        use_feature_augmentation=config['use_feature_augmentation'],
        use_prototype_refinement=config['use_prototype_refinement'],
        domain=config['domain']
    )
    
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Use model's parameter grouping method for robust separation
    param_groups = model.get_parameter_groups(lr_backbone_multiplier=0.1)
    
    # Create optimizer with parameter groups
    if params.optimization == 'Adam':
        optimizer = optim.Adam([
            {'params': pg['params'], 'lr': params.learning_rate * pg['lr_multiplier']}
            for pg in param_groups
        ], weight_decay=params.weight_decay)
    elif params.optimization == 'AdamW':
        optimizer = optim.AdamW([
            {'params': pg['params'], 'lr': params.learning_rate * pg['lr_multiplier']}
            for pg in param_groups
        ], weight_decay=params.weight_decay)
    elif params.optimization == 'SGD':
        optimizer = optim.SGD([
            {'params': pg['params'], 'lr': params.learning_rate * pg['lr_multiplier']}
            for pg in param_groups
        ], momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        raise ValueError(f'Unknown optimization: {params.optimization}')
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=params.num_epoch, eta_min=1e-6)
    
    # Training loop
    print(f"Training for {params.num_epoch} epochs...")
    
    best_acc = 0.0
    best_epoch = 0
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(
        './checkpoint_models',
        f'{params.dataset}_{params.backbone}_{params.k_shot}shot_enhanced'
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, params.num_epoch + 1):
        # Training
        train_loss, train_acc = train_epoch(
            model, base_loader, optimizer, epoch, params,
            grad_clip_value=params.grad_clip
        )
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{params.num_epoch}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Validation
        if epoch % params.val_freq == 0:
            val_acc, val_ci = evaluate(model, val_loader, params)
            print(f"  Val Acc: {val_acc:.4f} ± {val_ci:.4f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'params': params
                }
                
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.tar'))
                print(f"  ✓ Saved best model (acc: {best_acc:.4f})")
        
        # Save periodic checkpoint
        if epoch % params.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'params': params
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'{epoch}.tar'))
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    
    return model, best_acc


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Enhanced Few-Shot Model')
    
    # Dataset and model
    parser.add_argument('--dataset', default='miniImagenet', 
                       help='Dataset: omniglot/CUB/miniImagenet/HAM10000')
    parser.add_argument('--backbone', default='Conv4',
                       help='Backbone: Conv4/ResNet18/ResNet34')
    parser.add_argument('--n_way', default=5, type=int,
                       help='Number of classes per episode')
    parser.add_argument('--k_shot', default=1, type=int,
                       help='Number of support samples per class')
    parser.add_argument('--n_query', default=16, type=int,
                       help='Number of query samples per class')
    
    # Model architecture
    parser.add_argument('--feature_dim', default=64, type=int,
                       help='Feature dimension for transformer')
    parser.add_argument('--n_heads', default=4, type=int,
                       help='Number of attention heads')
    parser.add_argument('--dropout', default=0.1, type=float,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--num_epoch', default=100, type=int,
                       help='Number of training epochs')
    parser.add_argument('--n_episode', default=100, type=int,
                       help='Number of episodes per epoch')
    parser.add_argument('--test_iter', default=600, type=int,
                       help='Number of episodes for validation')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                       help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                       help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float,
                       help='Momentum for SGD')
    parser.add_argument('--optimization', default='AdamW',
                       help='Optimizer: Adam/AdamW/SGD')
    parser.add_argument('--grad_clip', default=1.0, type=float,
                       help='Gradient clipping value')
    parser.add_argument('--train_aug', default=0, type=int,
                       help='Use data augmentation during training')
    
    # Checkpointing
    parser.add_argument('--save_freq', default=25, type=int,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--val_freq', default=5, type=int,
                       help='Validate every N epochs')
    
    params = parser.parse_args()
    
    print("=" * 80)
    print("Enhanced Few-Shot Learning Training")
    print("=" * 80)
    print(f"Dataset: {params.dataset}")
    print(f"Backbone: {params.backbone}")
    print(f"Task: {params.n_way}-way {params.k_shot}-shot")
    print(f"Optimization: {params.optimization}")
    print(f"Learning rate: {params.learning_rate}")
    print("=" * 80)
    
    # Train model
    model, best_acc = train_enhanced_model(params)
    
    print("\nTraining finished successfully!")


if __name__ == '__main__':
    main()
