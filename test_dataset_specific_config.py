#!/usr/bin/env python3
"""
Test script to validate dataset-specific configurations
"""

import torch
import numpy as np
from methods.transformer import FewShotTransformer, Attention


def test_dataset_specific_attention():
    """Test that Attention module uses dataset-specific parameters"""
    print("Testing dataset-specific attention parameters...")
    
    # Test CUB dataset
    attn_cub = Attention(
        dim=512, heads=16, dim_head=96, variant='cosine',
        initial_cov_weight=0.65, initial_var_weight=0.15,
        dynamic_weight=True, n_way=5, k_shot=5,
        dropout=0.1, dataset='CUB'
    )
    
    # Test Yoga dataset
    attn_yoga = Attention(
        dim=512, heads=14, dim_head=88, variant='cosine',
        initial_cov_weight=0.6, initial_var_weight=0.25,
        dynamic_weight=True, n_way=5, k_shot=5,
        dropout=0.12, dataset='Yoga'
    )
    
    # Test general dataset (miniImageNet)
    attn_general = Attention(
        dim=512, heads=12, dim_head=80, variant='cosine',
        initial_cov_weight=0.55, initial_var_weight=0.2,
        dynamic_weight=True, n_way=5, k_shot=5,
        dropout=0.15, dataset='miniImagenet'
    )
    
    # Verify temperature initialization
    assert torch.allclose(attn_cub.temperature.data, torch.tensor(0.3), atol=0.01), \
        f"CUB temperature should be 0.3, got {attn_cub.temperature.data[0]}"
    assert torch.allclose(attn_yoga.temperature.data, torch.tensor(0.3), atol=0.01), \
        f"Yoga temperature should be 0.3, got {attn_yoga.temperature.data[0]}"
    assert torch.allclose(attn_general.temperature.data, torch.tensor(0.4), atol=0.01), \
        f"General temperature should be 0.4, got {attn_general.temperature.data[0]}"
    
    # Verify EMA decay
    assert attn_cub.ema_decay == 0.985, f"CUB EMA decay should be 0.985, got {attn_cub.ema_decay}"
    assert attn_yoga.ema_decay == 0.985, f"Yoga EMA decay should be 0.985, got {attn_yoga.ema_decay}"
    assert attn_general.ema_decay == 0.98, f"General EMA decay should be 0.98, got {attn_general.ema_decay}"
    
    # Verify gamma schedules
    assert attn_cub.gamma_start == 0.7, f"CUB gamma_start should be 0.7, got {attn_cub.gamma_start}"
    assert attn_cub.gamma_end == 0.02, f"CUB gamma_end should be 0.02, got {attn_cub.gamma_end}"
    
    assert attn_yoga.gamma_start == 0.65, f"Yoga gamma_start should be 0.65, got {attn_yoga.gamma_start}"
    assert attn_yoga.gamma_end == 0.025, f"Yoga gamma_end should be 0.025, got {attn_yoga.gamma_end}"
    
    assert attn_general.gamma_start == 0.6, f"General gamma_start should be 0.6, got {attn_general.gamma_start}"
    assert attn_general.gamma_end == 0.03, f"General gamma_end should be 0.03, got {attn_general.gamma_end}"
    
    print("✅ All dataset-specific attention parameters are correctly configured!")
    return True


def test_model_architecture():
    """Test that model architectures differ by dataset"""
    print("\nTesting dataset-specific model architectures...")
    
    # Mock feature model
    class MockFeatureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.final_feat_dim = 512
            self.conv1 = torch.nn.Conv2d(3, 64, 3)
        
        def forward(self, x):
            return torch.randn(x.size(0), 512)
    
    def mock_model_func():
        return MockFeatureModel()
    
    # Create models for different datasets
    model_cub = FewShotTransformer(
        mock_model_func, n_way=5, k_shot=5, n_query=15,
        variant='cosine', depth=2, heads=16, dim_head=96,
        mlp_dim=1024, initial_cov_weight=0.65,
        initial_var_weight=0.15, dynamic_weight=True,
        label_smoothing=0.05, attention_dropout=0.1,
        drop_path_rate=0.05, dataset='CUB'
    )
    
    model_yoga = FewShotTransformer(
        mock_model_func, n_way=5, k_shot=5, n_query=15,
        variant='cosine', depth=2, heads=14, dim_head=88,
        mlp_dim=896, initial_cov_weight=0.6,
        initial_var_weight=0.25, dynamic_weight=True,
        label_smoothing=0.08, attention_dropout=0.12,
        drop_path_rate=0.08, dataset='Yoga'
    )
    
    model_mini = FewShotTransformer(
        mock_model_func, n_way=5, k_shot=5, n_query=15,
        variant='cosine', depth=2, heads=12, dim_head=80,
        mlp_dim=768, initial_cov_weight=0.55,
        initial_var_weight=0.2, dynamic_weight=True,
        label_smoothing=0.1, attention_dropout=0.15,
        drop_path_rate=0.1, dataset='miniImagenet'
    )
    
    # Verify heads
    assert model_cub.ATTN.heads == 16, f"CUB should have 16 heads, got {model_cub.ATTN.heads}"
    assert model_yoga.ATTN.heads == 14, f"Yoga should have 14 heads, got {model_yoga.ATTN.heads}"
    assert model_mini.ATTN.heads == 12, f"MiniImageNet should have 12 heads, got {model_mini.ATTN.heads}"
    
    # Verify dataset attribute
    assert model_cub.dataset == 'CUB', f"CUB dataset attribute not set"
    assert model_yoga.dataset == 'Yoga', f"Yoga dataset attribute not set"
    assert model_mini.dataset == 'miniImagenet', f"MiniImageNet dataset attribute not set"
    
    # Verify label smoothing
    assert model_cub.loss_fn.label_smoothing == 0.05, f"CUB label smoothing should be 0.05"
    assert model_yoga.loss_fn.label_smoothing == 0.08, f"Yoga label smoothing should be 0.08"
    assert model_mini.loss_fn.label_smoothing == 0.1, f"MiniImageNet label smoothing should be 0.1"
    
    print("✅ All dataset-specific model architectures are correctly configured!")
    return True


def test_forward_pass():
    """Test that forward pass works for all dataset configurations"""
    print("\nTesting forward pass for all datasets...")
    
    class MockFeatureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.final_feat_dim = 512
            self.conv1 = torch.nn.Conv2d(3, 64, 3)
        
        def forward(self, x):
            batch_size = x.size(0)
            return torch.randn(batch_size, 512)
    
    def mock_model_func():
        return MockFeatureModel()
    
    datasets = ['CUB', 'Yoga', 'miniImagenet']
    configs = {
        'CUB': {'heads': 16, 'dim_head': 96, 'mlp_dim': 1024},
        'Yoga': {'heads': 14, 'dim_head': 88, 'mlp_dim': 896},
        'miniImagenet': {'heads': 12, 'dim_head': 80, 'mlp_dim': 768}
    }
    
    for dataset in datasets:
        config = configs[dataset]
        model = FewShotTransformer(
            mock_model_func, n_way=5, k_shot=5, n_query=15,
            variant='cosine', depth=2,
            heads=config['heads'], dim_head=config['dim_head'],
            mlp_dim=config['mlp_dim'], dynamic_weight=True,
            dataset=dataset
        )
        
        # Create dummy input: [n_way, k_shot + n_query, C, H, W]
        x = torch.randn(5, 20, 3, 84, 84)
        
        try:
            model.eval()
            with torch.no_grad():
                scores = model.set_forward(x, is_feature=False)
            
            # Check output shape
            expected_shape = (5 * 15, 5)  # (n_way * n_query, n_way)
            assert scores.shape == expected_shape, \
                f"Expected output shape {expected_shape}, got {scores.shape}"
            
            print(f"  ✅ {dataset}: Forward pass successful, output shape {scores.shape}")
        
        except Exception as e:
            print(f"  ❌ {dataset}: Forward pass failed with error: {e}")
            return False
    
    print("✅ All forward passes successful!")
    return True


def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing Dataset-Specific Configuration")
    print("=" * 70)
    
    all_passed = True
    
    try:
        all_passed = test_dataset_specific_attention() and all_passed
    except Exception as e:
        print(f"❌ Attention test failed: {e}")
        all_passed = False
    
    try:
        all_passed = test_model_architecture() and all_passed
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
        all_passed = False
    
    try:
        all_passed = test_forward_pass() and all_passed
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nDataset-specific configurations:")
        print("  • CUB: 16 heads, 96 dim_head, temp=0.3, gamma=0.7→0.02")
        print("  • Yoga: 14 heads, 88 dim_head, temp=0.3, gamma=0.65→0.025")
        print("  • miniImageNet/CIFAR: 12 heads, 80 dim_head, temp=0.4, gamma=0.6→0.03")
    else:
        print("❌ SOME TESTS FAILED")
        return 1
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())
