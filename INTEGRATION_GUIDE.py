"""
Integration Guide: Using OptimalFewShotModel with Existing Training Scripts

This guide shows how to integrate the OptimalFewShotModel into the existing
training infrastructure (train.py, train_test.py).
"""

# ============================================================================
# METHOD 1: Add to io_utils.py model dictionary
# ============================================================================

# In io_utils.py, add to model_dict:
"""
from methods.optimal_fewshot import OptimalFewShotModel

def get_optimal_fewshot_model(params):
    '''Helper to create OptimalFewShotModel with proper params'''
    from methods.optimal_fewshot import DATASET_CONFIGS
    
    # Get dataset config
    dataset_name = params.dataset
    if dataset_name not in DATASET_CONFIGS:
        print(f"Warning: {dataset_name} not in predefined configs. Using defaults.")
        config = {
            'feature_dim': 64,
            'n_heads': 4,
            'dropout': 0.1,
            'lr_backbone': 0.0005
        }
    else:
        config = DATASET_CONFIGS[dataset_name]
    
    # Determine if using custom backbone
    use_custom = (params.backbone == 'Conv4')
    
    # Create model_func if not using custom backbone
    if use_custom:
        model_func = None
    else:
        if params.backbone == 'Conv6':
            model_func = lambda: backbone.Conv6(params.dataset, flatten=True)
        elif params.backbone == 'ResNet12':
            model_func = lambda: backbone.ResNet12(params.FETI, params.dataset, flatten=True)
        elif params.backbone == 'ResNet18':
            model_func = lambda: backbone.ResNet18(params.FETI, params.dataset, flatten=True)
        elif params.backbone == 'ResNet34':
            model_func = lambda: backbone.ResNet34(params.FETI, params.dataset, flatten=True)
        else:
            model_func = lambda: backbone.Conv4(params.dataset, flatten=True)
    
    # Create model
    model = OptimalFewShotModel(
        model_func=model_func,
        n_way=params.n_way,
        k_shot=params.k_shot,
        n_query=params.n_query,
        feature_dim=config['feature_dim'],
        n_heads=config['n_heads'],
        dropout=config.get('dropout', 0.1),
        num_datasets=5,
        dataset=params.dataset,
        gradient_checkpointing=True,
        use_custom_backbone=use_custom
    )
    
    return model

# Add to model_dict
model_dict = {
    # ... existing methods ...
    'OptimalFewShot': get_optimal_fewshot_model,
}
"""


# ============================================================================
# METHOD 2: Direct usage in training script
# ============================================================================

"""
# In train.py or train_test.py:

import torch
import torch.optim as optim
from methods.optimal_fewshot import OptimalFewShotModel, DATASET_CONFIGS

def train_optimal_fewshot(params):
    '''Train OptimalFewShotModel'''
    
    # Get dataset configuration
    config = DATASET_CONFIGS.get(params.dataset, DATASET_CONFIGS['miniImagenet'])
    
    # Create data loaders
    from data.datamgr import SetDataManager
    train_loader = SetDataManager(
        params.dataset, 
        params.n_way, 
        params.k_shot, 
        params.n_query,
        n_episode=params.n_episode
    ).get_data_loader()
    
    val_loader = SetDataManager(
        params.dataset, 
        params.n_way, 
        params.k_shot, 
        params.n_query,
        n_episode=100  # Fixed validation episodes
    ).get_data_loader()
    
    # Create model
    model = OptimalFewShotModel(
        model_func=None,  # Use custom Conv4
        n_way=params.n_way,
        k_shot=params.k_shot,
        n_query=params.n_query,
        feature_dim=config['feature_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        num_datasets=5,
        dataset=params.dataset,
        gradient_checkpointing=True,
        use_custom_backbone=True
    ).cuda()
    
    # Create optimizer with dataset-specific learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr_backbone'],
        weight_decay=5e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=params.num_epoch,
        eta_min=config['lr_backbone'] * 0.01
    )
    
    # Mixed precision training (optional but recommended)
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    # Training loop
    max_acc = 0
    for epoch in range(params.num_epoch):
        # Training
        model.train()
        avg_loss = 0
        avg_acc = []
        
        for i, (x, _) in enumerate(train_loader):
            x = x.cuda()
            
            # Use mixed precision
            with autocast():
                acc, loss = model.set_forward_loss(x)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            avg_loss += loss.item()
            avg_acc.append(acc)
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.cuda()
                correct_this, count_this = model.correct(x)
                correct += correct_this
                total += count_this
        
        val_acc = (correct / total) * 100
        print(f'Epoch {epoch+1}/{params.num_epoch} | '
              f'Train Acc: {np.mean(avg_acc)*100:.2f}% | '
              f'Train Loss: {avg_loss/len(train_loader):.4f} | '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'accuracy': val_acc
            }, f'checkpoint_models/optimal_fewshot_{params.dataset}_best.pth')
            print(f'Best model saved! Accuracy: {val_acc:.2f}%')
    
    print(f'Training complete! Best validation accuracy: {max_acc:.2f}%')
    print(f'Target accuracy for {params.dataset}: {config["target_5shot"]*100:.1f}%')
    
    return max_acc

# Usage:
# python train.py --method OptimalFewShot --dataset miniImagenet --n_way 5 --k_shot 5
"""


# ============================================================================
# METHOD 3: Command-line training example
# ============================================================================

"""
# Basic training on miniImageNet
python train.py \\
    --method OptimalFewShot \\
    --dataset miniImagenet \\
    --backbone Conv4 \\
    --n_way 5 \\
    --k_shot 5 \\
    --n_query 15 \\
    --num_epoch 100 \\
    --n_episode 600

# Training on CUB with higher dropout
python train.py \\
    --method OptimalFewShot \\
    --dataset CUB \\
    --backbone Conv4 \\
    --n_way 5 \\
    --k_shot 5 \\
    --n_query 15 \\
    --num_epoch 100 \\
    --dropout 0.15

# Training on HAM10000 with focal loss (automatic)
python train.py \\
    --method OptimalFewShot \\
    --dataset HAM10000 \\
    --backbone Conv4 \\
    --n_way 7 \\
    --k_shot 5 \\
    --n_query 15 \\
    --num_epoch 150

# Training on Omniglot (1-shot)
python train.py \\
    --method OptimalFewShot \\
    --dataset Omniglot \\
    --backbone Conv4 \\
    --n_way 5 \\
    --k_shot 1 \\
    --n_query 15 \\
    --num_epoch 50
"""


# ============================================================================
# METHOD 4: Loading pre-trained model for inference
# ============================================================================

"""
import torch
from methods.optimal_fewshot import OptimalFewShotModel, DATASET_CONFIGS

def load_optimal_model(checkpoint_path, dataset='miniImagenet'):
    '''Load pre-trained OptimalFewShotModel'''
    
    # Get configuration
    config = DATASET_CONFIGS[dataset]
    
    # Create model
    model = OptimalFewShotModel(
        model_func=None,
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        n_query=15,
        feature_dim=config['feature_dim'],
        n_heads=config['n_heads'],
        dropout=0.0,  # No dropout during inference
        num_datasets=5,
        dataset=dataset,
        gradient_checkpointing=False,  # Not needed for inference
        use_custom_backbone=True
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model

# Usage:
# model = load_optimal_model('checkpoint_models/optimal_fewshot_miniImagenet_best.pth')
# logits, vic_loss, info = model.set_forward(test_episode)
"""


# ============================================================================
# METHOD 5: Custom dataset integration
# ============================================================================

"""
from methods.optimal_fewshot import DATASET_CONFIGS

# Add your custom dataset configuration
DATASET_CONFIGS['MyDataset'] = {
    'n_way': 5,
    'k_shot': 5,
    'input_size': 84,  # Image size
    'lr_backbone': 0.001,
    'dropout': 0.1,
    'target_5shot': 0.70,  # Target accuracy
    'dataset_id': 5,  # Unique ID
    'feature_dim': 64,
    'n_heads': 4
}

# Update dataset_id_map in OptimalFewShotModel
# In methods/optimal_fewshot.py, line ~330:
self.dataset_id_map = {
    'Omniglot': 0,
    'CUB': 1,
    'CIFAR': 2,
    'miniImagenet': 3,
    'HAM10000': 4,
    'MyDataset': 5  # Add your dataset
}

# Then use as normal:
# python train.py --method OptimalFewShot --dataset MyDataset --n_way 5 --k_shot 5
"""


# ============================================================================
# TIPS AND BEST PRACTICES
# ============================================================================

"""
1. **Memory Management**:
   - Always use gradient_checkpointing=True for training
   - Enable mixed precision (autocast + GradScaler) for 50% memory saving
   - If still OOM, reduce feature_dim to 32 or n_heads to 2

2. **Learning Rate**:
   - Use dataset-specific lr from DATASET_CONFIGS
   - Warmup for first 5-10 epochs
   - Cosine annealing for smooth decay

3. **Monitoring**:
   - Watch lambda_var (should increase 0.1→0.2)
   - Watch lambda_cov (should decrease 0.03→0.01)
   - Temperature should stay 8-15
   - VIC losses should decrease

4. **Hyperparameter Tuning**:
   - Start with default configs
   - Adjust dropout based on overfitting (0.05-0.2)
   - Increase epochs if accuracy plateaus early
   - Use focal_loss for imbalanced datasets (HAM10000)

5. **Validation**:
   - Run test_optimal_fewshot.py before training
   - Monitor validation accuracy every epoch
   - Save best model based on validation
   - Test on multiple seeds for robust results

6. **Debugging**:
   - Check input shapes match expected (n_way, k_shot+n_query, C, H, W)
   - Ensure images are normalized properly
   - Monitor gradients (should not be NaN or too large)
   - Check that model.training is True during training
"""

print("Integration guide ready!")
print("See comments above for different integration methods.")
