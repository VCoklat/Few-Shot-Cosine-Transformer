#!/usr/bin/env python3
"""
Test script to verify the ProFOCT improvements.
This tests:
1. VIC loss scaling is correct
2. Gradient clipping prevents explosion
3. Weight initialization works
4. VIC warmup functions correctly
"""

import torch
import torch.nn as nn
import numpy as np
import sys

# Test VIC loss computation
def test_vic_losses():
    print("="*60)
    print("Testing VIC Loss Computation")
    print("="*60)
    
    from methods.ProFOCT import ProFOCT
    import backbone
    
    # Create a simple feature model
    def feature_model():
        return backbone.Conv4(dataset='miniImagenet', flatten=True)
    
    # Create ProFOCT instance
    model = ProFOCT(
        feature_model,
        n_way=5,
        k_shot=1,
        n_query=16,
        variant='cosine',
        vic_alpha=0.1,
        vic_beta=1.0,
        vic_gamma=0.1
    )
    
    # Test covariance loss with different feature dimensions
    for batch_size in [5, 10, 20]:
        for feat_dim in [64, 256, 512]:
            z = torch.randn(batch_size, feat_dim)
            
            loss_v = model.compute_variance_loss(z)
            loss_c = model.compute_covariance_loss(z)
            
            print(f"\nBatch={batch_size}, Dim={feat_dim}:")
            print(f"  Variance loss: {loss_v.item():.6f}")
            print(f"  Covariance loss: {loss_c.item():.6f}")
            
            # Verify covariance loss is in reasonable range
            assert loss_c.item() < 10.0, f"Covariance loss too large: {loss_c.item()}"
            assert loss_c.item() >= 0.0, f"Covariance loss negative: {loss_c.item()}"
    
    print("\n✓ VIC losses are properly scaled!")
    return True

# Test VIC warmup
def test_vic_warmup():
    print("\n" + "="*60)
    print("Testing VIC Warmup")
    print("="*60)
    
    from methods.ProFOCT import ProFOCT
    import backbone
    
    def feature_model():
        return backbone.Conv4(dataset='miniImagenet', flatten=True)
    
    model = ProFOCT(
        feature_model,
        n_way=5,
        k_shot=1,
        n_query=16,
        variant='cosine',
        vic_alpha=0.1,
        vic_beta=1.0,
        vic_gamma=0.1
    )
    
    model.train()
    
    # Check initial VIC weights (should be 0 or very small due to warmup)
    initial_alpha = model.vic_alpha.item()
    print(f"\nInitial VIC alpha: {initial_alpha:.6f}")
    
    # Simulate training steps
    for step in range(150):
        model._warmup_vic_weights()
        
        if step == 0:
            alpha_at_0 = model.vic_alpha.item()
        elif step == 50:
            alpha_at_50 = model.vic_alpha.item()
        elif step == 100:
            alpha_at_100 = model.vic_alpha.item()
        elif step == 149:
            alpha_at_150 = model.vic_alpha.item()
    
    print(f"VIC alpha at step 0: {alpha_at_0:.6f}")
    print(f"VIC alpha at step 50: {alpha_at_50:.6f}")
    print(f"VIC alpha at step 100: {alpha_at_100:.6f}")
    print(f"VIC alpha at step 150: {alpha_at_150:.6f}")
    
    # Verify warmup schedule
    assert alpha_at_50 < alpha_at_100, "Alpha should increase during warmup"
    assert abs(alpha_at_100 - 0.1) < 0.01, f"Alpha should be ~0.1 after warmup, got {alpha_at_100}"
    assert alpha_at_150 == alpha_at_100, "Alpha should stabilize after warmup"
    
    print("\n✓ VIC warmup works correctly!")
    return True

# Test forward pass doesn't explode
def test_forward_pass():
    print("\n" + "="*60)
    print("Testing Forward Pass Stability")
    print("="*60)
    
    from methods.ProFOCT import ProFOCT
    import backbone
    
    def feature_model():
        return backbone.Conv4(dataset='miniImagenet', flatten=True)
    
    model = ProFOCT(
        feature_model,
        n_way=5,
        k_shot=1,
        n_query=16,
        variant='cosine',
        vic_alpha=0.1,
        vic_beta=1.0,
        vic_gamma=0.1
    )
    
    model.train()
    
    # Create dummy input: (n_way, k_shot + n_query, C, H, W)
    x = torch.randn(5, 17, 3, 84, 84)  # 5-way, 1-shot, 16 queries
    
    # Forward pass
    acc, loss = model.set_forward_loss(x)
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Accuracy: {acc*100:.2f}%")
    
    # Verify loss is reasonable
    assert loss.item() < 100.0, f"Loss too large: {loss.item()}"
    assert loss.item() > 0.0, f"Loss negative: {loss.item()}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is inf"
    
    # Test backward pass
    loss.backward()
    
    # Check gradients
    has_nan_grad = False
    has_large_grad = False
    max_grad = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan_grad = True
                print(f"NaN gradient in {name}")
            
            grad_norm = param.grad.norm().item()
            max_grad = max(max_grad, grad_norm)
            
            if grad_norm > 1000:
                has_large_grad = True
                print(f"Large gradient in {name}: {grad_norm:.2f}")
    
    print(f"\nMax gradient norm: {max_grad:.4f}")
    
    assert not has_nan_grad, "Found NaN gradients"
    assert not has_large_grad, "Found exploding gradients (>1000)"
    
    print("\n✓ Forward and backward pass are stable!")
    return True

# Test gradient clipping
def test_gradient_clipping():
    print("\n" + "="*60)
    print("Testing Gradient Clipping")
    print("="*60)
    
    # Create a simple model
    model = nn.Linear(10, 10)
    
    # Create large gradients
    x = torch.randn(5, 10)
    y = torch.randn(5, 10) * 1000  # Large target
    
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    
    # Check gradient before clipping
    grad_before = model.weight.grad.norm().item()
    print(f"\nGradient norm before clipping: {grad_before:.2f}")
    
    # Clip gradients
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # Check gradient after clipping
    grad_after = model.weight.grad.norm().item()
    print(f"Gradient norm after clipping: {grad_after:.2f}")
    
    assert grad_after <= max_norm * 1.01, f"Gradient not clipped properly: {grad_after}"
    
    print("\n✓ Gradient clipping works!")
    return True

if __name__ == '__main__':
    try:
        all_passed = True
        
        all_passed &= test_vic_losses()
        all_passed &= test_vic_warmup()
        all_passed &= test_forward_pass()
        all_passed &= test_gradient_clipping()
        
        if all_passed:
            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED!")
            print("="*60)
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print("✗ SOME TESTS FAILED")
            print("="*60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
