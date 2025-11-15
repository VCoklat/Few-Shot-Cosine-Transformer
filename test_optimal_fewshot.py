"""
Unit tests for the Optimal Few-Shot Learning Algorithm

Tests all components:
- SEBlock
- OptimizedConv4
- CosineAttention
- LightweightCosineTransformer
- DynamicVICRegularizer
- EpisodeAdaptiveLambda
- OptimalFewShotModel
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.optimal_fewshot import (
    SEBlock,
    OptimizedConv4,
    CosineAttention,
    LightweightCosineTransformer,
    DynamicVICRegularizer,
    EpisodeAdaptiveLambda,
    OptimalFewShotModel,
    DATASET_CONFIGS,
    focal_loss
)


def test_seblock():
    """Test SEBlock channel attention"""
    print("Testing SEBlock...")
    block = SEBlock(channel=64, reduction=4)
    x = torch.randn(2, 64, 10, 10)
    out = block(x)
    assert out.shape == x.shape, f"SEBlock output shape mismatch: {out.shape} vs {x.shape}"
    print("✓ SEBlock test passed")


def test_optimized_conv4():
    """Test OptimizedConv4 backbone"""
    print("Testing OptimizedConv4...")
    
    # Test with miniImagenet (3 channels)
    backbone = OptimizedConv4(hid_dim=64, dropout=0.1, dataset='miniImagenet')
    x = torch.randn(4, 3, 84, 84)
    out = backbone(x)
    assert out.shape == (4, 64), f"Conv4 output shape mismatch: {out.shape}"
    assert torch.allclose(torch.norm(out, p=2, dim=1), torch.ones(4), atol=1e-5), "Output not normalized"
    print("✓ OptimizedConv4 (RGB) test passed")
    
    # Test with Omniglot (1 channel)
    backbone = OptimizedConv4(hid_dim=64, dropout=0.1, dataset='Omniglot')
    x = torch.randn(4, 1, 28, 28)
    out = backbone(x)
    assert out.shape == (4, 64), f"Conv4 output shape mismatch: {out.shape}"
    print("✓ OptimizedConv4 (Grayscale) test passed")


def test_cosine_attention():
    """Test CosineAttention mechanism"""
    print("Testing CosineAttention...")
    attn = CosineAttention(dim=64, temperature=0.05)
    q = torch.randn(2, 4, 10, 16)  # (batch, heads, seq_len, dim_head)
    k = torch.randn(2, 4, 10, 16)
    v = torch.randn(2, 4, 10, 16)
    out, attn_weights = attn(q, k, v)
    assert out.shape == v.shape, f"Attention output shape mismatch: {out.shape} vs {v.shape}"
    assert attn_weights.shape == (2, 4, 10, 10), f"Attention weights shape mismatch: {attn_weights.shape}"
    print("✓ CosineAttention test passed")


def test_lightweight_transformer():
    """Test LightweightCosineTransformer"""
    print("Testing LightweightCosineTransformer...")
    transformer = LightweightCosineTransformer(d_model=64, n_heads=4, dropout=0.1)
    x = torch.randn(2, 20, 64)  # (batch, seq_len, dim)
    out = transformer(x)
    assert out.shape == x.shape, f"Transformer output shape mismatch: {out.shape} vs {x.shape}"
    print("✓ LightweightCosineTransformer test passed")


def test_dynamic_vic_regularizer():
    """Test DynamicVICRegularizer"""
    print("Testing DynamicVICRegularizer...")
    vic = DynamicVICRegularizer(feature_dim=64)
    
    # Test with multiple prototypes
    prototypes = torch.randn(5, 64)
    support_features = torch.randn(25, 64)
    lambda_var = torch.tensor(0.1)
    lambda_cov = torch.tensor(0.01)
    
    vic_loss, info = vic(prototypes, support_features, lambda_var, lambda_cov)
    assert isinstance(vic_loss, torch.Tensor), "VIC loss should be a tensor"
    assert 'var_loss' in info and 'cov_loss' in info, "Info dict should contain loss components"
    print("✓ DynamicVICRegularizer test passed")


def test_episode_adaptive_lambda():
    """Test EpisodeAdaptiveLambda predictor"""
    print("Testing EpisodeAdaptiveLambda...")
    predictor = EpisodeAdaptiveLambda(feature_dim=64, num_datasets=5)
    
    prototypes = torch.randn(5, 64)
    support_features = torch.randn(25, 64)
    query_features = torch.randn(75, 64)
    
    lambda_var, lambda_cov = predictor(prototypes, support_features, query_features, dataset_id=0)
    
    assert isinstance(lambda_var, torch.Tensor), "lambda_var should be a tensor"
    assert isinstance(lambda_cov, torch.Tensor), "lambda_cov should be a tensor"
    assert 0.05 <= lambda_var.item() <= 0.3, f"lambda_var out of range: {lambda_var.item()}"
    assert 0.005 <= lambda_cov.item() <= 0.1, f"lambda_cov out of range: {lambda_cov.item()}"
    print("✓ EpisodeAdaptiveLambda test passed")


def test_optimal_fewshot_model():
    """Test complete OptimalFewShotModel"""
    print("Testing OptimalFewShotModel...")
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    # Create model
    model = OptimalFewShotModel(
        model_func=None,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        feature_dim=64,
        n_heads=4,
        dropout=0.1,
        num_datasets=5,
        dataset='miniImagenet',
        gradient_checkpointing=False,  # Disable for testing
        use_custom_backbone=True
    )
    
    # Create dummy episode
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        logits, vic_loss, info = model.set_forward(x, is_feature=False)
    
    assert logits.shape == (n_way * n_query, n_way), f"Logits shape mismatch: {logits.shape}"
    assert isinstance(vic_loss, torch.Tensor), "VIC loss should be a tensor"
    assert 'lambda_var' in info, "Info should contain lambda_var"
    assert 'lambda_cov' in info, "Info should contain lambda_cov"
    assert 'temperature' in info, "Info should contain temperature"
    print("✓ OptimalFewShotModel forward test passed")
    
    # Test set_forward_loss
    model.train()
    acc, total_loss = model.set_forward_loss(x)
    assert 0 <= acc <= 1, f"Accuracy out of range: {acc}"
    assert isinstance(total_loss, torch.Tensor), "Total loss should be a tensor"
    print("✓ OptimalFewShotModel loss test passed")
    
    # Test correct method
    correct, total = model.correct(x)
    assert 0 <= correct <= total, f"Correct count out of range: {correct}/{total}"
    print("✓ OptimalFewShotModel correct test passed")


def test_dataset_configs():
    """Test dataset configurations"""
    print("Testing dataset configurations...")
    
    required_keys = ['n_way', 'k_shot', 'input_size', 'lr_backbone', 'dropout', 
                     'target_5shot', 'dataset_id', 'feature_dim', 'n_heads']
    
    for dataset_name, config in DATASET_CONFIGS.items():
        for key in required_keys:
            assert key in config, f"Config for {dataset_name} missing key: {key}"
        print(f"✓ Config for {dataset_name} validated")


def test_focal_loss():
    """Test focal loss function"""
    print("Testing focal_loss...")
    
    logits = torch.randn(10, 5)
    labels = torch.randint(0, 5, (10,))
    
    loss = focal_loss(logits, labels, alpha=0.25, gamma=2.0)
    assert isinstance(loss, torch.Tensor), "Focal loss should be a tensor"
    assert loss.item() > 0, "Focal loss should be positive"
    print("✓ focal_loss test passed")


def test_memory_efficiency():
    """Test memory efficiency features"""
    print("Testing memory efficiency...")
    
    # Create model with gradient checkpointing
    model = OptimalFewShotModel(
        model_func=None,
        n_way=5,
        k_shot=5,
        n_query=15,
        feature_dim=64,
        n_heads=4,
        dropout=0.1,
        num_datasets=5,
        dataset='miniImagenet',
        gradient_checkpointing=True,
        use_custom_backbone=True
    )
    
    # Check that gradient checkpointing is enabled
    assert model.gradient_checkpointing, "Gradient checkpointing should be enabled"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Check bias-free convolutions
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    bias_free = all(m.bias is None for m in conv_layers)
    assert bias_free, "All Conv2d layers should be bias-free"
    print("✓ Memory efficiency test passed")


def test_different_datasets():
    """Test model with different dataset configurations"""
    print("Testing different datasets...")
    
    datasets = ['Omniglot', 'CUB', 'CIFAR', 'miniImagenet', 'HAM10000']
    
    for dataset_name in datasets:
        config = DATASET_CONFIGS[dataset_name]
        
        # Create model
        model = OptimalFewShotModel(
            model_func=None,
            n_way=config['n_way'],
            k_shot=config['k_shot'],
            n_query=15,
            feature_dim=config['feature_dim'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
            num_datasets=5,
            dataset=dataset_name,
            gradient_checkpointing=False,
            use_custom_backbone=True
        )
        
        # Create appropriate input
        in_channels = 1 if dataset_name == 'Omniglot' else 3
        x = torch.randn(config['n_way'], config['k_shot'] + 15, 
                       in_channels, config['input_size'], config['input_size'])
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits, vic_loss, info = model.set_forward(x, is_feature=False)
        
        assert logits.shape[0] == config['n_way'] * 15, f"Logits shape mismatch for {dataset_name}"
        print(f"✓ {dataset_name} test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("Running Optimal Few-Shot Learning Algorithm Tests")
    print("=" * 80)
    
    tests = [
        test_seblock,
        test_optimized_conv4,
        test_cosine_attention,
        test_lightweight_transformer,
        test_dynamic_vic_regularizer,
        test_episode_adaptive_lambda,
        test_optimal_fewshot_model,
        test_dataset_configs,
        test_focal_loss,
        test_memory_efficiency,
        test_different_datasets,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print()
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
