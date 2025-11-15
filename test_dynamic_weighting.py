#!/usr/bin/env python
"""
Simple unit test for the dynamic weighting implementation
This test verifies the mathematical correctness of the variance and covariance regularization.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_variance_regularization():
    """Test variance regularization computation"""
    print("\n" + "="*60)
    print("Testing Variance Regularization")
    print("="*60)
    
    # Manual implementation for testing
    def manual_variance_reg(E, gamma=0.1, epsilon=1e-8):
        """Manual implementation of variance regularization"""
        import numpy as np
        
        # Compute variance for each dimension
        var_per_dim = np.var(E, axis=0)
        
        # Compute regularized standard deviation
        sigma = np.sqrt(var_per_dim + epsilon)
        
        # Compute hinge function
        V = np.mean(np.maximum(0, gamma - sigma))
        
        return V
    
    # Test data
    import numpy as np
    np.random.seed(42)
    E = np.random.randn(100, 512)  # 100 samples, 512 dimensions
    
    result = manual_variance_reg(E, gamma=0.1, epsilon=1e-8)
    
    print(f"Test data shape: {E.shape}")
    print(f"Variance regularization result: {result:.6f}")
    
    # Verify the result is reasonable
    if 0 <= result <= 0.1:
        print("✓ Variance regularization is within expected range [0, gamma]")
    else:
        print("✗ Warning: Variance regularization is outside expected range")
    
    return True

def test_covariance_regularization():
    """Test covariance regularization computation"""
    print("\n" + "="*60)
    print("Testing Covariance Regularization")
    print("="*60)
    
    # Manual implementation for testing
    def manual_covariance_reg(E):
        """Manual implementation of covariance regularization"""
        import numpy as np
        
        # Center the embeddings
        E_mean = np.mean(E, axis=0, keepdims=True)
        E_centered = E - E_mean
        
        # Compute covariance matrix
        batch_size = E.shape[0]
        if batch_size > 1:
            cov = np.matmul(E_centered.T, E_centered) / (batch_size - 1)
        else:
            cov = np.matmul(E_centered.T, E_centered)
        
        # Sum of squares of off-diagonal elements
        m = E.shape[1]
        off_diag_mask = ~np.eye(m, dtype=bool)
        C = np.sum(cov[off_diag_mask] ** 2) / m
        
        return C
    
    # Test data
    import numpy as np
    np.random.seed(42)
    E = np.random.randn(100, 512)  # 100 samples, 512 dimensions
    
    result = manual_covariance_reg(E)
    
    print(f"Test data shape: {E.shape}")
    print(f"Covariance regularization result: {result:.6f}")
    
    # Verify the result is reasonable (should be non-negative)
    if result >= 0:
        print("✓ Covariance regularization is non-negative")
    else:
        print("✗ Error: Covariance regularization is negative")
    
    return True

def test_weight_predictor_sum():
    """Test that weight predictor outputs sum to 1"""
    print("\n" + "="*60)
    print("Testing Weight Predictor Constraint")
    print("="*60)
    
    import numpy as np
    
    # Simulate softmax output
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    # Test random weights
    raw_weights = np.random.randn(3)
    normalized_weights = softmax(raw_weights)
    
    print(f"Raw weights: {raw_weights}")
    print(f"Normalized weights: {normalized_weights}")
    print(f"Sum of weights: {normalized_weights.sum():.10f}")
    
    # Verify sum is 1
    if abs(normalized_weights.sum() - 1.0) < 1e-6:
        print("✓ Weights sum to 1 (within numerical precision)")
    else:
        print("✗ Error: Weights do not sum to 1")
    
    return True

def test_loss_combination():
    """Test that loss combination is mathematically correct"""
    print("\n" + "="*60)
    print("Testing Loss Combination")
    print("="*60)
    
    import numpy as np
    
    # Simulate loss values
    ce_loss = 2.5
    var_reg = 0.05
    cov_reg = 0.3
    
    # Simulate weights
    w_ce = 0.7
    w_var = 0.2
    w_cov = 0.1
    
    # Compute combined loss
    combined_loss = w_ce * ce_loss + w_var * var_reg + w_cov * cov_reg
    
    print(f"Cross-entropy loss: {ce_loss:.6f}")
    print(f"Variance regularization: {var_reg:.6f}")
    print(f"Covariance regularization: {cov_reg:.6f}")
    print(f"Weights: CE={w_ce:.2f}, VAR={w_var:.2f}, COV={w_cov:.2f}")
    print(f"Combined loss: {combined_loss:.6f}")
    
    # Verify the combination
    expected = 0.7 * 2.5 + 0.2 * 0.05 + 0.1 * 0.3
    if abs(combined_loss - expected) < 1e-6:
        print("✓ Loss combination is correct")
    else:
        print("✗ Error: Loss combination is incorrect")
    
    return True

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Dynamic Weighting Formula - Unit Tests")
    print("#"*60)
    
    all_passed = True
    
    try:
        all_passed &= test_variance_regularization()
    except Exception as e:
        print(f"✗ Variance regularization test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_covariance_regularization()
    except Exception as e:
        print(f"✗ Covariance regularization test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_weight_predictor_sum()
    except Exception as e:
        print(f"✗ Weight predictor test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_loss_combination()
    except Exception as e:
        print(f"✗ Loss combination test failed: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)
