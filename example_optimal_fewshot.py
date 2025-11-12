"""
Example usage of the Optimal Few-Shot Learning Algorithm

This script demonstrates how to use the OptimalFewShotModel with:
- SE-Enhanced Conv4 backbone
- Lightweight Cosine Transformer
- Dynamic VIC Regularization
- Episode-Adaptive Lambda Predictor

Usage:
    python example_optimal_fewshot.py --dataset miniImagenet --n_way 5 --k_shot 5
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from methods.optimal_fewshot import (
    OptimalFewShotModel, 
    DATASET_CONFIGS, 
    focal_loss
)
import backbone


def get_model_config(dataset_name):
    """Get configuration for specific dataset"""
    if dataset_name not in DATASET_CONFIGS:
        print(f"Warning: Dataset {dataset_name} not in predefined configs. Using miniImagenet config.")
        dataset_name = 'miniImagenet'
    
    return DATASET_CONFIGS[dataset_name]


def create_optimal_model(dataset_name, n_way=5, k_shot=5, n_query=15, 
                        use_custom_backbone=True, gradient_checkpointing=True):
    """
    Create an OptimalFewShotModel instance
    
    Args:
        dataset_name: Name of the dataset ('Omniglot', 'CUB', 'CIFAR', 'miniImagenet', 'HAM10000')
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        use_custom_backbone: Use optimized Conv4 backbone (True) or provided backbone (False)
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    
    Returns:
        model: OptimalFewShotModel instance
        config: Dataset configuration
    """
    config = get_model_config(dataset_name)
    
    # Create dummy model_func (not used if use_custom_backbone=True)
    if use_custom_backbone:
        model_func = None  # Will be ignored
    else:
        # Use Conv4 from backbone.py
        model_func = lambda: backbone.Conv4(dataset_name, flatten=True)
    
    # Create model
    model = OptimalFewShotModel(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        feature_dim=config['feature_dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        num_datasets=5,
        dataset=dataset_name,
        gradient_checkpointing=gradient_checkpointing,
        use_custom_backbone=use_custom_backbone
    )
    
    return model, config


def create_dummy_episode(n_way, k_shot, n_query, input_size=84, in_channels=3):
    """Create dummy episode for testing
    
    Returns:
        x: Episode batch of shape (n_way, k_shot + n_query, C, H, W)
    """
    # Create episode with proper shape
    x = torch.randn(n_way, k_shot + n_query, in_channels, input_size, input_size)
    
    return x


def train_step_example(model, optimizer, x, use_focal_loss=False):
    """
    Example training step
    
    Args:
        model: OptimalFewShotModel instance
        optimizer: PyTorch optimizer
        x: Episode batch (support + query images)
        use_focal_loss: Use focal loss instead of cross-entropy
    
    Returns:
        acc: Accuracy
        loss: Total loss
        info: Additional information
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits, vic_loss, info = model.set_forward(x)
    
    # Get targets
    n_way = model.n_way
    n_query = model.n_query
    targets = torch.from_numpy(np.repeat(range(n_way), n_query))
    targets = targets.to(logits.device)
    
    # Classification loss
    if use_focal_loss:
        ce_loss = focal_loss(logits, targets)
    else:
        ce_loss = nn.functional.cross_entropy(logits, targets, label_smoothing=0.1)
    
    # Total loss
    total_loss = ce_loss + vic_loss
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    
    # Compute accuracy
    predict = torch.argmax(logits, dim=1)
    acc = (predict == targets).sum().item() / targets.size(0)
    
    return acc, total_loss.item(), info


def main():
    parser = argparse.ArgumentParser(description='Optimal Few-Shot Learning Example')
    parser.add_argument('--dataset', type=str, default='miniImagenet', 
                       choices=['Omniglot', 'CUB', 'CIFAR', 'miniImagenet', 'HAM10000'],
                       help='Dataset name')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support samples per class')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--use_custom_backbone', action='store_true', default=True,
                       help='Use optimized Conv4 backbone')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                       help='Enable gradient checkpointing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Optimal Few-Shot Learning Algorithm Example")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"N-way: {args.n_way}")
    print(f"K-shot: {args.k_shot}")
    print(f"N-query: {args.n_query}")
    print(f"Device: {args.device}")
    print(f"Custom Backbone: {args.use_custom_backbone}")
    print(f"Gradient Checkpointing: {args.gradient_checkpointing}")
    print("=" * 80)
    
    # Create model
    print("\nCreating model...")
    model, config = create_optimal_model(
        dataset_name=args.dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        use_custom_backbone=args.use_custom_backbone,
        gradient_checkpointing=args.gradient_checkpointing
    )
    model = model.to(args.device)
    
    # Print model info
    print(f"Model created successfully!")
    print(f"Feature dimension: {model.feat_dim}")
    print(f"Target accuracy (5-shot): {config['target_5shot']:.1%}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    print(f"\nCreating optimizer with lr={config['lr_backbone']}...")
    optimizer = optim.AdamW(model.parameters(), lr=config['lr_backbone'], weight_decay=5e-4)
    
    # Determine input size and channels
    input_size = config['input_size']
    in_channels = 1 if args.dataset == 'Omniglot' else 3
    
    # Run training episodes
    print(f"\nRunning {args.num_episodes} training episodes...")
    print("-" * 80)
    
    use_focal = config.get('focal_loss', False)
    if use_focal:
        print("Using focal loss for class imbalance")
    
    for episode in range(args.num_episodes):
        # Create dummy episode
        x = create_dummy_episode(
            args.n_way, args.k_shot, args.n_query, 
            input_size=input_size, in_channels=in_channels
        ).to(args.device)
        
        # Training step
        acc, loss, info = train_step_example(model, optimizer, x, use_focal_loss=use_focal)
        
        print(f"Episode {episode + 1}/{args.num_episodes} | "
              f"Acc: {acc:.4f} | Loss: {loss:.4f} | "
              f"λ_var: {info['lambda_var']:.4f} | λ_cov: {info['lambda_cov']:.4f} | "
              f"Temp: {info['temperature']:.2f}")
    
    print("-" * 80)
    print("\n✅ Example completed successfully!")
    print("\nKey Features Demonstrated:")
    print("  ✓ SE-Enhanced Conv4 backbone with channel attention")
    print("  ✓ Lightweight Cosine Transformer (single-layer, 4-head)")
    print("  ✓ Dynamic VIC Regularization (variance + covariance)")
    print("  ✓ Episode-Adaptive Lambda Predictor with EMA smoothing")
    print("  ✓ Gradient checkpointing for memory efficiency")
    print("  ✓ Mixed precision training support (optional)")
    print("\nMemory Optimizations:")
    print(f"  • Gradient checkpointing: {'Enabled' if args.gradient_checkpointing else 'Disabled'}")
    print("  • Bias-free convolutions: Enabled")
    print("  • Episode-wise training: Batch size = 1")
    print("  • Expected VRAM usage: ~3.5-4.5GB with mixed precision")
    print("\nFor full training, integrate with your dataset loaders and")
    print("use the provided train.py or train_test.py scripts.")
    print("=" * 80)


if __name__ == '__main__':
    main()
