#!/usr/bin/env python3
"""
Test script for OptimalFewShotModel
Verifies basic functionality and memory usage
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.optimal_few_shot import (
    OptimalFewShotModel, 
    SEBlock, 
    OptimizedConv4,
    LightweightCosineTransformer,
    DynamicVICRegularizer,
    EpisodeAdaptiveLambda,
    DATASET_CONFIGS
)

def test_se_block():
    """Test SEBlock functionality"""
    print("Testing SEBlock...")
    se = SEBlock(64, reduction=4)
    x = torch.randn(4, 64, 21, 21)
    out = se(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"✓ SEBlock test passed. Input shape: {x.shape}, Output shape: {out.shape}")

def test_optimized_conv4():
    """Test OptimizedConv4 backbone"""
    print("\nTesting OptimizedConv4...")
    for dataset in ['miniImagenet', 'CIFAR', 'Omniglot']:
        conv4 = OptimizedConv4(hid_dim=64, dropout=0.1, dataset=dataset)
        
        # Test with appropriate input
        if dataset == 'Omniglot':
            x = torch.randn(4, 3, 84, 84)  # Will be converted to single channel
        elif dataset == 'CIFAR':
            x = torch.randn(4, 3, 32, 32)
        else:
            x = torch.randn(4, 3, 84, 84)
        
        out = conv4(x)
        print(f"  Dataset: {dataset}, Input: {x.shape}, Output: {out.shape}, Final dim: {conv4.final_feat_dim}")
        assert out.shape[0] == 4, f"Batch size mismatch"
        assert len(out.shape) == 2, f"Output should be 2D"
    print("✓ OptimizedConv4 test passed")

def test_cosine_transformer():
    """Test LightweightCosineTransformer"""
    print("\nTesting LightweightCosineTransformer...")
    transformer = LightweightCosineTransformer(d_model=64, n_heads=4, dropout=0.1)
    x = torch.randn(2, 10, 64)  # Batch=2, Seq=10, Dim=64
    out = transformer(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"✓ Transformer test passed. Input shape: {x.shape}, Output shape: {out.shape}")

def test_vic_regularizer():
    """Test DynamicVICRegularizer"""
    print("\nTesting DynamicVICRegularizer...")
    vic = DynamicVICRegularizer(feature_dim=64)
    prototypes = torch.randn(5, 64)  # 5-way
    support_features = torch.randn(25, 64)  # 5-way, 5-shot
    
    vic_loss, vic_info = vic(prototypes, support_features, lambda_var=0.1, lambda_cov=0.01)
    print(f"  VIC Loss: {vic_loss.item():.4f}")
    print(f"  Var Loss: {vic_info['var_loss']:.4f}")
    print(f"  Cov Loss: {vic_info['cov_loss']:.4f}")
    print("✓ VIC Regularizer test passed")

def test_lambda_predictor():
    """Test EpisodeAdaptiveLambda"""
    print("\nTesting EpisodeAdaptiveLambda...")
    predictor = EpisodeAdaptiveLambda(feature_dim=64, num_datasets=5)
    prototypes = torch.randn(5, 64)
    support_features = torch.randn(25, 64)
    query_features = torch.randn(75, 64)
    
    lambda_var, lambda_cov = predictor(prototypes, support_features, query_features, dataset_id=0)
    print(f"  Lambda Var: {lambda_var.item():.4f}")
    print(f"  Lambda Cov: {lambda_cov.item():.4f}")
    print("✓ Lambda Predictor test passed")

def test_complete_model():
    """Test complete OptimalFewShotModel"""
    print("\nTesting Complete OptimalFewShotModel...")
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def dummy_model_func():
        return None
    
    model = OptimalFewShotModel(
        dummy_model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        feature_dim=64,
        n_heads=4,
        dropout=0.1,
        num_datasets=5,
        dataset='miniImagenet',
        use_focal_loss=False,
        label_smoothing=0.1
    )
    
    # Create dummy episode data
    # Shape: (n_way, k_shot + n_query, C, H, W)
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    print(f"  Input shape: {x.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        logits = model.set_forward(x)
        # Test internal method that returns full tuple
        logits_full, prototypes, support_features, query_features = model._set_forward_full(x)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Prototypes shape: {prototypes.shape}")
    print(f"  Support features shape: {support_features.shape}")
    print(f"  Query features shape: {query_features.shape}")
    
    assert logits.shape == (n_way * n_query, n_way), f"Logits shape mismatch: {logits.shape}"
    assert torch.allclose(logits, logits_full), "Logits from set_forward and _set_forward_full should match"
    assert prototypes.shape == (n_way, 64), f"Prototypes shape mismatch: {prototypes.shape}"
    
    # Test loss computation
    acc, loss = model.set_forward_loss(x)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Loss: {loss.item():.4f}")
    
    print("✓ Complete model test passed")

def test_memory_usage():
    """Test memory usage"""
    print("\nTesting Memory Usage...")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        return
    
    device = torch.device('cuda:0')
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def dummy_model_func():
        return None
    
    model = OptimalFewShotModel(
        dummy_model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        feature_dim=64,
        n_heads=4,
        dropout=0.1,
        num_datasets=5,
        dataset='miniImagenet',
        use_focal_loss=False,
        label_smoothing=0.1
    ).to(device)
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Create dummy episode data
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model.set_forward(x)
    
    # Get memory stats
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    print(f"  Current allocated: {allocated:.2f} GB")
    print(f"  Peak allocated: {peak:.2f} GB")
    
    if peak < 8.0:
        print(f"✓ Memory usage test passed (Peak: {peak:.2f} GB < 8 GB)")
    else:
        print(f"⚠ Warning: Peak memory usage {peak:.2f} GB exceeds 8 GB target")
    
    # Clean up
    del model, x, logits
    torch.cuda.empty_cache()

def test_dataset_configs():
    """Test dataset configurations"""
    print("\nTesting Dataset Configurations...")
    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"  {dataset_name}: n_way={config['n_way']}, k_shot={config['k_shot']}, "
              f"dropout={config['dropout']}, lr={config['lr_backbone']}")
    print("✓ Dataset configs test passed")

def main():
    """Run all tests"""
    print("="*60)
    print("Testing Optimal Few-Shot Learning Implementation")
    print("="*60)
    
    try:
        test_se_block()
        test_optimized_conv4()
        test_cosine_transformer()
        test_vic_regularizer()
        test_lambda_predictor()
        test_complete_model()
        test_memory_usage()
        test_dataset_configs()
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        return 0
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ Test failed with error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
