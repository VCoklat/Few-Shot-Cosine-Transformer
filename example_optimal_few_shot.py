#!/usr/bin/env python3
"""
Example usage of OptimalFewShotModel

This script demonstrates how to:
1. Create an OptimalFewShotModel instance
2. Generate synthetic episode data
3. Run forward and backward passes
4. Monitor training metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.optimal_few_shot import OptimalFewShotModel, DATASET_CONFIGS

def create_synthetic_episode(n_way=5, k_shot=5, n_query=15, image_size=84):
    """
    Create a synthetic few-shot episode for demonstration
    
    Args:
        n_way: Number of classes
        k_shot: Number of support examples per class
        n_query: Number of query examples per class
        image_size: Size of input images
    
    Returns:
        Batch tensor of shape (n_way, k_shot + n_query, 3, image_size, image_size)
    """
    batch_size = n_way * (k_shot + n_query)
    x = torch.randn(n_way, k_shot + n_query, 3, image_size, image_size)
    return x

def main():
    """Main demonstration function"""
    print("=" * 70)
    print("OptimalFewShotModel Usage Example")
    print("=" * 70)
    
    # Configuration
    dataset = 'miniImagenet'
    n_way = 5
    k_shot = 5
    n_query = 15
    
    print(f"\nDataset: {dataset}")
    print(f"Configuration: {n_way}-way {k_shot}-shot")
    print(f"Query samples per class: {n_query}")
    
    # Get dataset-specific configuration
    config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS['miniImagenet'])
    print(f"\nDataset-specific hyperparameters:")
    print(f"  - Dropout: {config['dropout']}")
    print(f"  - Learning rate: {config['lr_backbone']}")
    print(f"  - Target 5-shot accuracy: {config['target_5shot']*100:.1f}%")
    
    # Create model
    print("\n" + "-" * 70)
    print("Creating OptimalFewShotModel...")
    print("-" * 70)
    
    def dummy_model_func():
        return None
    
    model = OptimalFewShotModel(
        dummy_model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        feature_dim=64,
        n_heads=4,
        dropout=config['dropout'],
        num_datasets=5,
        dataset=dataset,
        use_focal_loss=config.get('focal_loss', False),
        label_smoothing=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Create synthetic episode
    print("\n" + "-" * 70)
    print("Generating synthetic episode data...")
    print("-" * 70)
    
    x = create_synthetic_episode(n_way, k_shot, n_query, image_size=84)
    print(f"Episode shape: {x.shape}")
    print(f"  - {n_way} classes")
    print(f"  - {k_shot} support examples per class")
    print(f"  - {n_query} query examples per class")
    
    # Training mode demonstration
    print("\n" + "-" * 70)
    print("Training Mode - Forward + Backward Pass")
    print("-" * 70)
    
    model.train()
    
    # Forward pass with loss computation
    acc, loss = model.set_forward_loss(x)
    
    print(f"\nForward pass completed:")
    print(f"  - Accuracy: {acc*100:.2f}%")
    print(f"  - Total Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nBackward pass completed successfully!")
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for _ in model.parameters())
    print(f"Gradients computed: {has_grad}/{total_params_count} parameters")
    
    # Evaluation mode demonstration
    print("\n" + "-" * 70)
    print("Evaluation Mode - Inference")
    print("-" * 70)
    
    model.eval()
    
    with torch.no_grad():
        logits, prototypes, support_features, query_features = model._set_forward_full(x)
        
        print(f"\nInference completed:")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Prototypes shape: {prototypes.shape}")
        print(f"  - Support features shape: {support_features.shape}")
        print(f"  - Query features shape: {query_features.shape}")
        
        # Compute predictions
        pred = logits.argmax(dim=1)
        target = torch.from_numpy(np.repeat(range(n_way), n_query))
        acc = (pred == target).sum().item() / target.size(0)
        
        print(f"\nPrediction accuracy: {acc*100:.2f}%")
    
    # Component information
    print("\n" + "-" * 70)
    print("Model Components")
    print("-" * 70)
    
    print("\n1. Optimized Conv4 Backbone:")
    print(f"   - Input channels: 3 (RGB)")
    print(f"   - Hidden dimension: 64")
    print(f"   - SE blocks: 4 (one per conv block)")
    print(f"   - Output dimension: {model.feat_dim}")
    
    print("\n2. Projection Layer:")
    print(f"   - Input: {model.feat_dim} → Output: 64")
    
    print("\n3. Cosine Transformer:")
    print(f"   - Layers: 1")
    print(f"   - Attention heads: 4")
    print(f"   - Dimension per head: 16")
    print(f"   - Gradient checkpointing: Enabled")
    
    print("\n4. VIC Regularizer:")
    print(f"   - Variance loss: Maximizes inter-class separation")
    print(f"   - Covariance loss: Decorrelates feature dimensions")
    
    print("\n5. Lambda Predictor:")
    print(f"   - Dataset embeddings: 5 datasets")
    print(f"   - EMA momentum: 0.9")
    print(f"   - Adaptive lambda range: [0.05-0.3] (var), [0.005-0.1] (cov)")
    
    print("\n6. Temperature Parameter:")
    print(f"   - Initial value: 10.0")
    print(f"   - Learnable: Yes")
    
    # Memory estimation (CPU only)
    print("\n" + "-" * 70)
    print("Memory Usage (CPU)")
    print("-" * 70)
    
    import sys
    model_size = sum(p.element_size() * p.numel() for p in model.parameters())
    print(f"Model size: {model_size / 1024**2:.2f} MB")
    
    # Usage recommendations
    print("\n" + "=" * 70)
    print("Usage Recommendations")
    print("=" * 70)
    
    print("\n1. Training command:")
    print(f"   python train_test.py \\")
    print(f"       --method OptimalFewShot \\")
    print(f"       --dataset {dataset} \\")
    print(f"       --n_way {n_way} \\")
    print(f"       --k_shot {k_shot} \\")
    print(f"       --num_epoch 100 \\")
    print(f"       --learning_rate {config['lr_backbone']} \\")
    print(f"       --wandb 1")
    
    print("\n2. Testing command:")
    print(f"   python train_test.py \\")
    print(f"       --method OptimalFewShot \\")
    print(f"       --dataset {dataset} \\")
    print(f"       --split novel \\")
    print(f"       --test_iter 600")
    
    print("\n3. Mixed precision training:")
    print("   The model automatically supports torch.cuda.amp for FP16 training")
    
    print("\n4. Gradient accumulation (if needed):")
    print("   Accumulate gradients over 2-4 steps for larger effective batch size")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
