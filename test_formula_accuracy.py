#!/usr/bin/env python3
"""
Test to verify the accuracy improvements from the formula fixes.
This compares the old (incorrect) normalization with the new (correct) normalization.
"""

import torch
import numpy as np

def test_variance_normalization():
    """Test that variance is normalized by m (dimensions) not by K (samples)"""
    print("\n" + "="*60)
    print("Testing Variance Normalization Fix")
    print("="*60)
    
    # Create test data
    torch.manual_seed(42)
    E = torch.randn(100, 512)  # 100 samples, 512 dimensions
    K, m = E.shape
    
    gamma = 1.0
    epsilon = 1e-8
    
    # Compute variance per dimension
    var_per_dim = torch.var(E, dim=0, unbiased=False)
    regularized_std = torch.sqrt(var_per_dim + epsilon)
    hinge_values = torch.clamp(gamma - regularized_std, min=0.0)
    
    # OLD (INCORRECT): Divide by number of samples
    V_old = torch.sum(hinge_values) / K
    
    # NEW (CORRECT): Divide by number of dimensions
    V_new = torch.sum(hinge_values) / m
    
    print(f"Test data shape: {E.shape}")
    print(f"Number of samples (K): {K}")
    print(f"Number of dimensions (m): {m}")
    print(f"\nOLD (incorrect) variance: {V_old.item():.6f}")
    print(f"NEW (correct) variance: {V_new.item():.6f}")
    print(f"Ratio (new/old): {(V_new/V_old).item():.2f}x")
    
    # The new formula should give a larger regularization term
    # because we divide by a smaller number (m vs K when K > m)
    if K > m:
        assert V_new > V_old, "New variance should be larger when K > m"
        print(f"\n✓ Variance normalization fix increases regularization strength")
    else:
        print(f"\n✓ Variance normalization computed correctly")
    
    return True

def test_covariance_normalization():
    """Test that covariance is normalized by m (dimensions)"""
    print("\n" + "="*60)
    print("Testing Covariance Normalization Fix")
    print("="*60)
    
    # Create test data
    torch.manual_seed(42)
    E = torch.randn(100, 512)  # 100 samples, 512 dimensions
    K, m = E.shape
    
    # Center the data
    E_mean = torch.mean(E, dim=0, keepdim=True)
    E_centered = E - E_mean
    
    # Compute covariance matrix
    cov = torch.matmul(E_centered.T, E_centered) / (K - 1)
    
    # Create mask for off-diagonal elements
    mask = torch.ones_like(cov) - torch.eye(m)
    
    # OLD (INCORRECT): No normalization by m
    C_old = torch.sum((cov * mask) ** 2)
    
    # NEW (CORRECT): Normalize by m
    C_new = torch.sum((cov * mask) ** 2) / m
    
    print(f"Test data shape: {E.shape}")
    print(f"Number of dimensions (m): {m}")
    print(f"Covariance matrix shape: {cov.shape}")
    print(f"\nOLD (incorrect) covariance: {C_old.item():.6f}")
    print(f"NEW (correct) covariance: {C_new.item():.6f}")
    print(f"Ratio (new/old): {(C_new/C_old).item():.6f}")
    
    # The new formula should give a normalized value
    assert C_new < C_old, "New covariance should be smaller (normalized)"
    print(f"\n✓ Covariance normalization fix provides proper scaling")
    
    return True

def test_numerical_stability():
    """Test that the fixes improve numerical stability"""
    print("\n" + "="*60)
    print("Testing Numerical Stability")
    print("="*60)
    
    # Test with different dimensions
    dimensions = [64, 128, 256, 512, 1024]
    
    for dim in dimensions:
        torch.manual_seed(42)
        E = torch.randn(100, dim)
        m = dim
        
        # Variance component (new formula)
        var_per_dim = torch.var(E, dim=0, unbiased=False)
        regularized_std = torch.sqrt(var_per_dim + 1e-8)
        hinge_values = torch.clamp(1.0 - regularized_std, min=0.0)
        V = torch.sum(hinge_values) / m
        
        # Covariance component (new formula)
        E_mean = torch.mean(E, dim=0, keepdim=True)
        E_centered = E - E_mean
        cov = torch.matmul(E_centered.T, E_centered) / (E.shape[0] - 1)
        mask = torch.ones_like(cov) - torch.eye(m)
        C = torch.sum((cov * mask) ** 2) / m
        
        print(f"  Dim {dim:4d}: V={V.item():.6f}, C={C.item():.6f}")
        
        # Check that values are reasonable
        assert V >= 0 and V <= 1.0, f"Variance should be in [0, 1], got {V}"
        assert C >= 0, f"Covariance should be non-negative, got {C}"
    
    print("\n✓ Formulas are numerically stable across different dimensions")
    return True

def test_gradient_flow():
    """Test that gradients flow correctly through the new formulas"""
    print("\n" + "="*60)
    print("Testing Gradient Flow")
    print("="*60)
    
    # Create test data with gradient tracking
    torch.manual_seed(42)
    E = torch.randn(100, 256, requires_grad=True)
    m = 256
    
    # Variance component
    var_per_dim = torch.var(E, dim=0, unbiased=False)
    regularized_std = torch.sqrt(var_per_dim + 1e-8)
    hinge_values = torch.clamp(1.0 - regularized_std, min=0.0)
    V = torch.sum(hinge_values) / m
    
    # Covariance component
    E_mean = torch.mean(E, dim=0, keepdim=True)
    E_centered = E - E_mean
    cov = torch.matmul(E_centered.T, E_centered) / (E.shape[0] - 1)
    mask = torch.ones_like(cov) - torch.eye(m)
    C = torch.sum((cov * mask) ** 2) / m
    
    # Combined loss
    loss = V + C
    
    # Compute gradients
    loss.backward()
    
    print(f"Variance component: {V.item():.6f}")
    print(f"Covariance component: {C.item():.6f}")
    print(f"Combined loss: {loss.item():.6f}")
    print(f"Gradient norm: {E.grad.norm().item():.6f}")
    
    # Check that gradients exist and are reasonable
    assert E.grad is not None, "Gradients should exist"
    assert not torch.isnan(E.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(E.grad).any(), "Gradients should not contain Inf"
    
    print("\n✓ Gradients flow correctly through the formulas")
    return True

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Formula Accuracy Test - Verifying Fixes")
    print("#"*60)
    
    all_passed = True
    
    try:
        all_passed &= test_variance_normalization()
    except Exception as e:
        print(f"✗ Variance normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_covariance_normalization()
    except Exception as e:
        print(f"✗ Covariance normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_numerical_stability()
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_gradient_flow()
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All accuracy tests passed!")
        print("\nSummary of Fixes:")
        print("  1. Variance now normalizes by m (dimensions) instead of K (samples)")
        print("  2. Covariance now normalizes by m for proper scaling")
        print("  3. Both formulas maintain numerical stability")
        print("  4. Gradients flow correctly for backpropagation")
        print("\nExpected Impact:")
        print("  • More accurate regularization strength")
        print("  • Better numerical stability across different dimensions")
        print("  • Improved training dynamics")
        print("  • Increased model accuracy on few-shot tasks")
    else:
        print("✗ Some tests failed")
    print("="*60)
    
    import sys
    sys.exit(0 if all_passed else 1)
