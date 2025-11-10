"""
Unit tests for VIC Regularization module
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import to avoid methods/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("vic_regularization", 
    os.path.join(os.path.dirname(__file__), "methods", "vic_regularization.py"))
vic_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vic_module)

VICRegularization = vic_module.VICRegularization
DynamicVICWeights = vic_module.DynamicVICWeights


def test_vic_regularization_basic():
    """Test basic VIC regularization functionality"""
    print("Testing VIC Regularization basic functionality...")
    
    # Create VIC regularization module
    vic_reg = VICRegularization(gamma=1.0, epsilon=1e-6)
    
    # Create sample embeddings (batch_size=10, embedding_dim=64)
    embeddings = torch.randn(10, 64)
    
    # Compute VIC losses
    vic_losses = vic_reg(embeddings)
    
    # Check that losses are computed
    assert 'variance_loss' in vic_losses, "Variance loss not computed"
    assert 'covariance_loss' in vic_losses, "Covariance loss not computed"
    
    # Check that losses are scalars
    assert vic_losses['variance_loss'].dim() == 0, "Variance loss should be scalar"
    assert vic_losses['covariance_loss'].dim() == 0, "Covariance loss should be scalar"
    
    # Check that losses are non-negative
    assert vic_losses['variance_loss'].item() >= 0, "Variance loss should be non-negative"
    assert vic_losses['covariance_loss'].item() >= 0, "Covariance loss should be non-negative"
    
    print(f"  Variance loss: {vic_losses['variance_loss'].item():.6f}")
    print(f"  Covariance loss: {vic_losses['covariance_loss'].item():.6f}")
    print("  ✓ VIC Regularization basic functionality test passed")


def test_variance_loss():
    """Test variance loss computation"""
    print("\nTesting variance loss computation...")
    
    vic_reg = VICRegularization(gamma=1.0, epsilon=1e-6)
    
    # Test with embeddings that have high variance (should have low loss)
    high_var_embeddings = torch.randn(100, 64) * 5  # High variance
    v_loss_high = vic_reg.variance_loss(high_var_embeddings)
    
    # Test with embeddings that have low variance (should have high loss)
    low_var_embeddings = torch.randn(100, 64) * 0.1  # Low variance
    v_loss_low = vic_reg.variance_loss(low_var_embeddings)
    
    print(f"  High variance loss: {v_loss_high.item():.6f}")
    print(f"  Low variance loss: {v_loss_low.item():.6f}")
    
    # Low variance embeddings should have higher variance loss
    assert v_loss_low.item() > v_loss_high.item(), \
        "Low variance embeddings should have higher variance loss"
    
    print("  ✓ Variance loss computation test passed")


def test_covariance_loss():
    """Test covariance loss computation"""
    print("\nTesting covariance loss computation...")
    
    vic_reg = VICRegularization(gamma=1.0, epsilon=1e-6)
    
    # Test with uncorrelated embeddings (should have low covariance loss)
    uncorrelated_embeddings = torch.randn(100, 64)
    c_loss_uncorr = vic_reg.covariance_loss(uncorrelated_embeddings)
    
    # Test with correlated embeddings (should have higher covariance loss)
    # Create embeddings where features are correlated
    base = torch.randn(100, 1)
    correlated_embeddings = base.repeat(1, 64) + torch.randn(100, 64) * 0.1
    c_loss_corr = vic_reg.covariance_loss(correlated_embeddings)
    
    print(f"  Uncorrelated covariance loss: {c_loss_uncorr.item():.6f}")
    print(f"  Correlated covariance loss: {c_loss_corr.item():.6f}")
    
    # Correlated embeddings should have higher covariance loss
    assert c_loss_corr.item() > c_loss_uncorr.item(), \
        "Correlated embeddings should have higher covariance loss"
    
    print("  ✓ Covariance loss computation test passed")


def test_dynamic_vic_weights():
    """Test dynamic VIC weight adjustment"""
    print("\nTesting dynamic VIC weight adjustment...")
    
    vic_weights = DynamicVICWeights(
        lambda_V_base=0.5,
        lambda_I=9.0,
        lambda_C_base=0.5
    )
    
    # Test at different epochs
    total_epochs = 50
    
    # Beginning of training
    weights_start = vic_weights.get_weights(0, total_epochs)
    print(f"  Epoch 0: λ_V={weights_start['lambda_V']:.3f}, "
          f"λ_I={weights_start['lambda_I']:.3f}, "
          f"λ_C={weights_start['lambda_C']:.3f}")
    
    # Middle of training
    weights_mid = vic_weights.get_weights(25, total_epochs)
    print(f"  Epoch 25: λ_V={weights_mid['lambda_V']:.3f}, "
          f"λ_I={weights_mid['lambda_I']:.3f}, "
          f"λ_C={weights_mid['lambda_C']:.3f}")
    
    # End of training
    weights_end = vic_weights.get_weights(49, total_epochs)
    print(f"  Epoch 49: λ_V={weights_end['lambda_V']:.3f}, "
          f"λ_I={weights_end['lambda_I']:.3f}, "
          f"λ_C={weights_end['lambda_C']:.3f}")
    
    # Check that variance weight increases
    assert weights_end['lambda_V'] > weights_start['lambda_V'], \
        "Variance weight should increase during training"
    
    # Check that invariance weight remains constant
    assert weights_start['lambda_I'] == weights_mid['lambda_I'] == weights_end['lambda_I'], \
        "Invariance weight should remain constant"
    
    # Check that covariance weight decreases
    assert weights_end['lambda_C'] < weights_start['lambda_C'], \
        "Covariance weight should decrease during training"
    
    print("  ✓ Dynamic VIC weight adjustment test passed")


def test_vic_gradient_flow():
    """Test that gradients flow correctly through VIC losses"""
    print("\nTesting gradient flow through VIC losses...")
    
    vic_reg = VICRegularization(gamma=1.0, epsilon=1e-6)
    
    # Create embeddings that require gradients
    embeddings = torch.randn(10, 64, requires_grad=True)
    
    # Compute VIC losses
    vic_losses = vic_reg(embeddings)
    
    # Compute combined loss
    total_loss = vic_losses['variance_loss'] + vic_losses['covariance_loss']
    
    # Backpropagate
    total_loss.backward()
    
    # Check that gradients are computed
    assert embeddings.grad is not None, "Gradients should be computed"
    assert not torch.allclose(embeddings.grad, torch.zeros_like(embeddings.grad)), \
        "Gradients should be non-zero"
    
    print(f"  Gradient norm: {embeddings.grad.norm().item():.6f}")
    print("  ✓ Gradient flow test passed")


def test_vic_with_different_batch_sizes():
    """Test VIC with different batch sizes"""
    print("\nTesting VIC with different batch sizes...")
    
    vic_reg = VICRegularization(gamma=1.0, epsilon=1e-6)
    
    batch_sizes = [5, 10, 50, 100]
    embedding_dim = 64
    
    for batch_size in batch_sizes:
        embeddings = torch.randn(batch_size, embedding_dim)
        vic_losses = vic_reg(embeddings)
        
        print(f"  Batch size {batch_size}: "
              f"V={vic_losses['variance_loss'].item():.4f}, "
              f"C={vic_losses['covariance_loss'].item():.4f}")
        
        # Check that losses are finite
        assert torch.isfinite(vic_losses['variance_loss']), \
            f"Variance loss should be finite for batch size {batch_size}"
        assert torch.isfinite(vic_losses['covariance_loss']), \
            f"Covariance loss should be finite for batch size {batch_size}"
    
    print("  ✓ Different batch sizes test passed")


def run_all_tests():
    """Run all VIC regularization tests"""
    print("="*60)
    print("Running VIC Regularization Tests")
    print("="*60)
    
    try:
        test_vic_regularization_basic()
        test_variance_loss()
        test_covariance_loss()
        test_dynamic_vic_weights()
        test_vic_gradient_flow()
        test_vic_with_different_batch_sizes()
        
        print("\n" + "="*60)
        print("✓ All VIC Regularization tests passed!")
        print("="*60)
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
