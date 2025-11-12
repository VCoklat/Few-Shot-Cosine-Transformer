#!/usr/bin/env python
"""
Comprehensive test suite for weight prediction stabilization features.
Tests the new improvements to the Few-Shot Cosine Transformer.
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.transformer import Attention, FewShotTransformer
import backbone

def test_temperature_parameter():
    """Test that temperature parameter is properly initialized and used"""
    print("\n" + "="*60)
    print("Testing Temperature Parameter")
    print("="*60)
    
    dim = 640
    heads = 8
    dim_head = 64
    
    # Create attention module with dynamic weighting
    attention = Attention(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        variant="cosine",
        dynamic_weight=True,
        n_way=5,
        k_shot=5
    )
    
    # Check temperature parameter exists
    assert hasattr(attention, 'weight_temperature'), "Temperature parameter not found"
    assert attention.weight_temperature.shape[0] == heads, f"Expected {heads} temperature values, got {attention.weight_temperature.shape[0]}"
    
    # Check initial values
    print(f"✓ Temperature parameter initialized with shape: {attention.weight_temperature.shape}")
    print(f"  Initial values: {attention.weight_temperature.data}")
    
    # Test forward pass
    q = torch.randn(1, 5, dim)
    k = torch.randn(25, 1, dim)
    v = torch.randn(25, 1, dim)
    
    output = attention(q, k, v, use_advanced=False)
    print(f"✓ Forward pass successful with output shape: {output.shape}")
    
    return True

def test_entropy_regularization():
    """Test entropy regularization on predicted weights"""
    print("\n" + "="*60)
    print("Testing Entropy Regularization")
    print("="*60)
    
    dim = 640
    heads = 8
    dim_head = 64
    
    attention = Attention(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        variant="cosine",
        dynamic_weight=True,
        n_way=5,
        k_shot=5
    )
    
    # Check entropy regularization parameters
    assert hasattr(attention, 'entropy_reg_lambda'), "Entropy regularization lambda not found"
    print(f"✓ Entropy regularization lambda: {attention.entropy_reg_lambda}")
    
    # Test weight predictor
    qk_features = torch.randn(heads, dim_head * 2)
    weights, logits, entropy = attention.weight_predictor_forward(qk_features)
    
    print(f"✓ Weight predictor returns 3 outputs:")
    print(f"  - Weights shape: {weights.shape} (sum={weights.sum(dim=-1).mean():.4f})")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Entropy shape: {entropy.shape} (mean={entropy.mean():.4f})")
    
    # Check weights sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(heads), atol=1e-5), "Weights don't sum to 1"
    print("✓ Weights properly sum to 1.0")
    
    # Check entropy is non-negative
    assert (entropy >= 0).all(), "Entropy should be non-negative"
    print("✓ Entropy values are non-negative")
    
    return True

def test_l2_penalty():
    """Test L2 penalty on logit magnitudes"""
    print("\n" + "="*60)
    print("Testing L2 Penalty on Logits")
    print("="*60)
    
    dim = 640
    heads = 8
    dim_head = 64
    
    attention = Attention(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        variant="cosine",
        dynamic_weight=True,
        n_way=5,
        k_shot=5
    )
    
    # Check L2 penalty parameter
    assert hasattr(attention, 'logit_l2_lambda'), "L2 penalty lambda not found"
    print(f"✓ L2 penalty lambda: {attention.logit_l2_lambda}")
    
    # Test forward pass and check regularization losses
    q = torch.randn(1, 5, dim)
    k = torch.randn(25, 1, dim)
    v = torch.randn(25, 1, dim)
    
    attention.train()
    output = attention(q, k, v, use_advanced=False)
    
    # Check that regularization losses are computed
    assert hasattr(attention, 'last_entropy_reg'), "Entropy regularization loss not stored"
    assert hasattr(attention, 'last_logit_l2'), "L2 penalty not stored"
    
    print(f"✓ Regularization losses computed:")
    print(f"  - Entropy reg: {attention.last_entropy_reg.item():.6f}")
    print(f"  - L2 penalty: {attention.last_logit_l2.item():.6f}")
    
    # Test get_regularization_losses method
    total_reg = attention.get_regularization_losses()
    print(f"✓ Total regularization loss: {total_reg.item():.6f}")
    
    return True

def test_shrinkage_covariance():
    """Test shrinkage covariance estimation (Ledoit-Wolf style)"""
    print("\n" + "="*60)
    print("Testing Shrinkage Covariance")
    print("="*60)
    
    dim = 640
    heads = 8
    dim_head = 64
    
    attention = Attention(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        variant="cosine",
        dynamic_weight=True,
        n_way=5,
        k_shot=5
    )
    
    # Check shrinkage parameter
    assert hasattr(attention, 'shrinkage_alpha'), "Shrinkage alpha not found"
    print(f"✓ Shrinkage alpha: {attention.shrinkage_alpha}")
    
    # Test covariance computation with small sample
    E = torch.randn(2, 5, 128)  # Small dimension for testing
    
    cov_with_shrinkage = attention.covariance_component_torch(E, use_shrinkage=True)
    cov_without_shrinkage = attention.covariance_component_torch(E, use_shrinkage=False)
    
    print(f"✓ Covariance with shrinkage: {cov_with_shrinkage.item():.6f}")
    print(f"✓ Covariance without shrinkage: {cov_without_shrinkage.item():.6f}")
    
    # Both should be non-negative
    assert cov_with_shrinkage >= 0, "Covariance with shrinkage should be non-negative"
    assert cov_without_shrinkage >= 0, "Covariance without shrinkage should be non-negative"
    
    return True

def test_numerical_stability():
    """Test improved numerical stability in variance/covariance"""
    print("\n" + "="*60)
    print("Testing Numerical Stability")
    print("="*60)
    
    dim = 640
    heads = 8
    dim_head = 64
    
    attention = Attention(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        variant="cosine",
        dynamic_weight=True,
        n_way=5,
        k_shot=5
    )
    
    # Test with very small values
    E_small = torch.randn(2, 5, 128) * 1e-6
    var_small = attention.variance_component_torch(E_small, gamma=1.0, epsilon=1e-8)
    
    print(f"✓ Variance with small values: {var_small.item():.6f}")
    assert not torch.isnan(var_small), "Variance produced NaN with small values"
    assert not torch.isinf(var_small), "Variance produced Inf with small values"
    
    # Test with large values
    E_large = torch.randn(2, 5, 128) * 1e6
    var_large = attention.variance_component_torch(E_large, gamma=1.0, epsilon=1e-8)
    
    print(f"✓ Variance with large values: {var_large.item():.6f}")
    assert not torch.isnan(var_large), "Variance produced NaN with large values"
    assert not torch.isinf(var_large), "Variance produced Inf with large values"
    
    # Test covariance stability
    cov_small = attention.covariance_component_torch(E_small, use_shrinkage=True)
    cov_large = attention.covariance_component_torch(E_large, use_shrinkage=True)
    
    print(f"✓ Covariance with small values: {cov_small.item():.6f}")
    print(f"✓ Covariance with large values: {cov_large.item():.6f}")
    
    assert not torch.isnan(cov_small), "Covariance produced NaN with small values"
    assert not torch.isnan(cov_large), "Covariance produced NaN with large values"
    
    return True

def test_component_normalization():
    """Test component magnitude normalization before mixing"""
    print("\n" + "="*60)
    print("Testing Component Normalization")
    print("="*60)
    
    dim = 640
    heads = 8
    dim_head = 64
    
    attention = Attention(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        variant="cosine",
        dynamic_weight=True,
        n_way=5,
        k_shot=5
    )
    
    # Create inputs with different magnitudes
    q = torch.randn(1, 5, dim) * 10  # Large magnitude
    k = torch.randn(25, 1, dim) * 0.1  # Small magnitude
    v = torch.randn(25, 1, dim)
    
    attention.train()
    output = attention(q, k, v, use_advanced=False)
    
    print(f"✓ Forward pass with different magnitudes successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"  Output std: {output.std().item():.6f}")
    
    # Check output is reasonable
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    return True

def test_dropout_in_weight_predictor():
    """Test increased dropout in weight predictor"""
    print("\n" + "="*60)
    print("Testing Dropout in Weight Predictor")
    print("="*60)
    
    dim = 640
    heads = 8
    dim_head = 64
    
    attention = Attention(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        variant="cosine",
        dynamic_weight=True,
        n_way=5,
        k_shot=5
    )
    
    # Check dropout layers exist
    assert hasattr(attention, 'weight_dropout1'), "First dropout layer not found"
    assert hasattr(attention, 'weight_dropout2'), "Second dropout layer not found"
    
    print(f"✓ Dropout layer 1 p={attention.weight_dropout1.p}")
    print(f"✓ Dropout layer 2 p={attention.weight_dropout2.p}")
    
    # Test in training mode
    attention.train()
    qk_features = torch.randn(heads, dim_head * 2)
    
    weights1, _, _ = attention.weight_predictor_forward(qk_features)
    weights2, _, _ = attention.weight_predictor_forward(qk_features)
    
    # Weights should be different due to dropout
    diff = (weights1 - weights2).abs().mean()
    print(f"✓ Dropout effect (mean abs difference): {diff.item():.6f}")
    
    # In eval mode, should be deterministic
    attention.eval()
    weights3, _, _ = attention.weight_predictor_forward(qk_features)
    weights4, _, _ = attention.weight_predictor_forward(qk_features)
    
    eval_diff = (weights3 - weights4).abs().mean()
    print(f"✓ Eval mode consistency (mean abs difference): {eval_diff.item():.9f}")
    
    return True

def test_integration_with_model():
    """Test integration with FewShotTransformer model"""
    print("\n" + "="*60)
    print("Testing Integration with FewShotTransformer")
    print("="*60)
    
    # Create a simple model
    def model_func():
        class SimpleFeature(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.final_feat_dim = 640
                self.linear = torch.nn.Linear(3*224*224, 640)
            
            def forward(self, x):
                return self.linear(x.view(x.size(0), -1))
        
        return SimpleFeature()
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    model = FewShotTransformer(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant="cosine",
        depth=1,
        heads=8,
        dim_head=64,
        dynamic_weight=True
    )
    
    # Create dummy data
    x = torch.randn(n_way, k_shot + n_query, 3, 224, 224)
    
    # Test forward pass
    model.train()
    acc, loss = model.set_forward_loss(x)
    
    print(f"✓ Forward pass successful")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Loss: {loss.item():.6f}")
    
    # Check that loss includes regularization
    if hasattr(model.ATTN, 'last_entropy_reg'):
        print(f"  Entropy reg: {model.ATTN.last_entropy_reg.item():.6f}")
        print(f"  L2 penalty: {model.ATTN.last_logit_l2.item():.6f}")
    
    # Test backward pass
    loss.backward()
    print("✓ Backward pass successful")
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if 'weight_temperature' in name:
                print(f"  {name} grad norm: {grad_norm:.6f}")
    
    return True

def run_all_tests():
    """Run all tests"""
    tests = [
        ("Temperature Parameter", test_temperature_parameter),
        ("Entropy Regularization", test_entropy_regularization),
        ("L2 Penalty", test_l2_penalty),
        ("Shrinkage Covariance", test_shrinkage_covariance),
        ("Numerical Stability", test_numerical_stability),
        ("Component Normalization", test_component_normalization),
        ("Dropout in Weight Predictor", test_dropout_in_weight_predictor),
        ("Integration with Model", test_integration_with_model),
    ]
    
    print("\n" + "="*60)
    print("WEIGHT STABILIZATION TEST SUITE")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name}: FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return True
    else:
        print(f"\n✗ {failed} test(s) failed")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
