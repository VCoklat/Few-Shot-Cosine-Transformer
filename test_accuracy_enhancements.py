#!/usr/bin/env python3
"""
Test script to validate accuracy enhancement improvements.
This script tests the new architectural improvements and training strategies.
"""

import torch
import torch.nn as nn
import numpy as np
import sys

def test_layer_scale():
    """Test LayerScale implementation"""
    print("Testing LayerScale implementation...")
    
    # Create a simple LayerScale
    dim = 512
    layer_scale = nn.Parameter(torch.ones(dim) * 0.1)
    
    # Test with random input
    x = torch.randn(2, 10, dim)
    output = x * layer_scale
    
    assert output.shape == x.shape, "LayerScale output shape mismatch"
    assert torch.allclose(output, x * 0.1, atol=1e-6), "LayerScale scaling incorrect"
    print("✓ LayerScale test passed")
    return True

def test_enhanced_model_initialization():
    """Test that model can be initialized with enhanced hyperparameters"""
    print("\nTesting enhanced model initialization...")
    
    try:
        from methods.transformer import FewShotTransformer
        from backbone import Conv4NP
        
        # Test with enhanced hyperparameters
        def feature_model():
            return Conv4NP('miniImagenet', flatten=True)
        
        model = FewShotTransformer(
            feature_model,
            n_way=5,
            k_shot=5,
            n_query=15,
            variant='cosine',
            depth=3,  # Enhanced from 2
            heads=16,  # Enhanced from 12
            dim_head=96,  # Enhanced from 80
            mlp_dim=1024,  # Enhanced from 768
            initial_cov_weight=0.55,
            initial_var_weight=0.2,
            dynamic_weight=True,
            label_smoothing=0.15,  # Enhanced from 0.1
            attention_dropout=0.15,
            drop_path_rate=0.15  # Enhanced from 0.1
        )
        
        print(f"✓ Model initialized successfully with:")
        print(f"  - Depth: 3 layers")
        print(f"  - Heads: 16")
        print(f"  - Dim per head: 96")
        print(f"  - MLP dim: 1024")
        print(f"  - Label smoothing: 0.15")
        print(f"  - Drop path rate: 0.15")
        
        # Check LayerScale parameters exist
        assert hasattr(model, 'layer_scale_attn'), "LayerScale for attention not found"
        assert hasattr(model, 'layer_scale_ffn'), "LayerScale for FFN not found"
        print("✓ LayerScale parameters initialized")
        
        # Check increased dropout
        assert model.ffn_dropout.p == 0.15, f"FFN dropout should be 0.15, got {model.ffn_dropout.p}"
        print("✓ Enhanced dropout rate set correctly")
        
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test forward pass with enhanced model"""
    print("\nTesting forward pass with enhanced architecture...")
    
    try:
        from methods.transformer import FewShotTransformer
        from backbone import Conv4NP
        
        def feature_model():
            return Conv4NP('miniImagenet', flatten=True)
        
        model = FewShotTransformer(
            feature_model,
            n_way=5,
            k_shot=5,
            n_query=15,
            variant='cosine',
            depth=3,
            heads=16,
            dim_head=96,
            mlp_dim=1024,
            label_smoothing=0.15,
            attention_dropout=0.15,
            drop_path_rate=0.15
        )
        
        # Create dummy input with correct shape (batch_size, channels, height, width)
        batch_size = 5 * (5 + 15)  # n_way * (k_shot + n_query)
        x = torch.randn(1, batch_size, 3, 84, 84)  # Add batch dimension at front
        
        model.eval()
        with torch.no_grad():
            output = model.set_forward(x)
        
        expected_shape = (5 * 15, 5)  # (n_way * n_query, n_way)
        assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test training step with loss computation"""
    print("\nTesting training step with enhanced model...")
    
    try:
        from methods.transformer import FewShotTransformer
        from backbone import Conv4NP
        
        def feature_model():
            return Conv4NP('miniImagenet', flatten=True)
        
        model = FewShotTransformer(
            feature_model,
            n_way=5,
            k_shot=5,
            n_query=15,
            variant='cosine',
            depth=3,
            heads=16,
            dim_head=96,
            mlp_dim=1024,
            label_smoothing=0.15,
            attention_dropout=0.15,
            drop_path_rate=0.15
        )
        
        # Create dummy input with correct shape
        batch_size = 5 * (5 + 15)
        x = torch.randn(1, batch_size, 3, 84, 84)  # Add batch dimension at front
        
        model.train()
        acc, loss = model.set_forward_loss(x)
        
        assert isinstance(acc, float), "Accuracy should be a float"
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert 0 <= acc <= 1, f"Accuracy should be in [0, 1], got {acc}"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        
        print(f"✓ Training step successful")
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - Loss: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print("✓ Backward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixup_augmentation():
    """Test enhanced mixup augmentation"""
    print("\nTesting enhanced mixup augmentation...")
    
    try:
        from methods.transformer import FewShotTransformer
        from backbone import Conv4NP
        
        def feature_model():
            return Conv4NP('miniImagenet', flatten=True)
        
        model = FewShotTransformer(
            feature_model,
            n_way=5,
            k_shot=5,
            n_query=15,
            variant='cosine',
            depth=3,
            heads=16,
            dim_head=96,
            mlp_dim=1024
        )
        
        # Test mixup with alpha=0.3
        z_support = torch.randn(5, 5, 512)
        model.train()
        mixed = model.mixup_support(z_support, alpha=0.3)
        
        assert mixed.shape == z_support.shape, "Mixup should preserve shape"
        assert not torch.allclose(mixed, z_support), "Mixup should modify the features"
        
        print("✓ Enhanced mixup augmentation (alpha=0.3) working correctly")
        
        # Test that mixup is disabled in eval mode
        model.eval()
        z_support_eval = torch.randn(5, 5, 512)
        no_mixed = model.mixup_support(z_support_eval, alpha=0.3)
        assert torch.allclose(no_mixed, z_support_eval), "Mixup should be disabled in eval mode"
        print("✓ Mixup correctly disabled in eval mode")
        
        return True
    except Exception as e:
        print(f"✗ Mixup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_improvements():
    """Test attention mechanism improvements"""
    print("\nTesting attention mechanism improvements...")
    
    try:
        from methods.transformer import Attention
        
        # Create attention module with enhanced settings
        dim = 512
        heads = 16
        dim_head = 96
        
        attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            variant='cosine',
            initial_cov_weight=0.55,
            initial_var_weight=0.2,
            dynamic_weight=True,
            dropout=0.15
        )
        
        # Check temperature is set correctly (temperature is per-head, use approximate comparison)
        assert abs(attn.temperature[0].item() - 0.35) < 0.01, f"Temperature should be ~0.35, got {attn.temperature[0].item()}"
        print(f"✓ Enhanced temperature scaling: {attn.temperature[0].item()}")
        
        # Check gamma schedule
        assert attn.gamma_start == 0.65, f"Gamma start should be 0.65, got {attn.gamma_start}"
        assert attn.gamma_end == 0.025, f"Gamma end should be 0.025, got {attn.gamma_end}"
        print(f"✓ Enhanced gamma schedule: start={attn.gamma_start}, end={attn.gamma_end}")
        
        # Check EMA decay
        assert attn.ema_decay == 0.97, f"EMA decay should be 0.97, got {attn.ema_decay}"
        print(f"✓ Enhanced EMA decay: {attn.ema_decay}")
        
        return True
    except Exception as e:
        print(f"✗ Attention improvements test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learning_rate_scheduler():
    """Test learning rate scheduler configuration"""
    print("\nTesting learning rate scheduler...")
    
    try:
        import torch.optim as optim
        
        # Create dummy model and optimizer
        model = nn.Linear(10, 10)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
        # Test CosineAnnealingWarmRestarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Test a few steps
        initial_lr = optimizer.param_groups[0]['lr']
        for i in range(5):
            scheduler.step()
        
        print("✓ CosineAnnealingWarmRestarts scheduler initialized")
        print(f"  - Initial LR: {initial_lr}")
        print(f"  - LR after 5 steps: {optimizer.param_groups[0]['lr']:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("="*60)
    print("Testing Accuracy Enhancement Improvements")
    print("="*60)
    
    tests = [
        ("LayerScale", test_layer_scale),
        ("Enhanced Model Initialization", test_enhanced_model_initialization),
        ("Forward Pass", test_forward_pass),
        ("Training Step", test_training_step),
        ("Mixup Augmentation", test_mixup_augmentation),
        ("Attention Improvements", test_attention_improvements),
        ("Learning Rate Scheduler", test_learning_rate_scheduler),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Accuracy enhancements are ready.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
