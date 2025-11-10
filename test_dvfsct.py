"""
Unit tests for Dynamic-VIC FS-CT (DV-FSCT) implementation.

This module contains tests for the key components of the DV-FSCT algorithm:
- VIC loss components (Variance, Invariance, Covariance)
- Dynamic weight calculation based on hardness
- Prototype computation
- Cosine attention mechanism
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.dvfsct import DVFSCT, CosineAttention
from backbone import Conv4

device = torch.device('cpu')  # Use CPU for testing


def test_vic_variance_loss():
    """Test variance loss computation"""
    print("\n=== Testing VIC Variance Loss ===")
    
    # Create dummy model
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16
    ).to(device)
    
    # Create random features
    Z = torch.randn(5, 5, 512)  # [N, K, d]
    
    # Compute variance loss
    V = model.vic_variance_loss(Z)
    
    print(f"Variance loss: {V.item():.4f}")
    assert V.item() >= 0, "Variance loss should be non-negative"
    print("✓ Variance loss computed successfully")


def test_vic_covariance_loss():
    """Test covariance loss computation"""
    print("\n=== Testing VIC Covariance Loss ===")
    
    # Create dummy model
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16
    ).to(device)
    
    # Create random features
    Z = torch.randn(5, 5, 512)  # [N, K, d]
    
    # Compute covariance loss
    C = model.vic_covariance_loss(Z)
    
    print(f"Covariance loss: {C.item():.4f}")
    assert C.item() >= 0, "Covariance loss should be non-negative"
    print("✓ Covariance loss computed successfully")


def test_hardness_computation():
    """Test dynamic hardness score computation"""
    print("\n=== Testing Hardness Score Computation ===")
    
    # Create dummy model
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16
    ).to(device)
    
    # Create features where some are far from mean (high hardness)
    Z_S = torch.randn(5, 5, 512)  # [N, K, d]
    
    # Compute hardness
    h_bar, h_classes = model.compute_hardness_scores(Z_S)
    
    print(f"Average hardness: {h_bar.item():.4f}")
    print(f"Class hardness: {h_classes.detach().cpu().numpy()}")
    
    assert 0 <= h_bar.item() <= 2, f"Hardness should be in reasonable range, got {h_bar.item()}"
    assert h_classes.shape[0] == 5, "Should have hardness for each class"
    print("✓ Hardness computation successful")


def test_prototype_computation():
    """Test learnable prototype computation"""
    print("\n=== Testing Prototype Computation ===")
    
    # Create dummy model
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16
    ).to(device)
    
    # Create random support features
    Z_S = torch.randn(5, 5, 512)  # [N, K, d]
    
    # Compute prototypes
    P = model.compute_prototypes(Z_S)
    
    print(f"Prototypes shape: {P.shape}")
    assert P.shape == (5, 512), f"Expected shape (5, 512), got {P.shape}"
    print("✓ Prototype computation successful")


def test_cosine_attention():
    """Test cosine attention mechanism"""
    print("\n=== Testing Cosine Attention ===")
    
    # Create attention layer
    attn = CosineAttention(dim=512, heads=8, dim_head=64).to(device)
    
    # Create dummy inputs (attention is from k/v to q)
    q = torch.randn(1, 5, 512)  # [batch_q, n_proto, dim]
    k = torch.randn(16, 1, 512)  # [batch_k, n_query, dim]
    v = torch.randn(16, 1, 512)  # [batch_v, n_query, dim]
    
    # Apply attention
    out = attn(q, k, v)
    
    print(f"Attention output shape: {out.shape}")
    # Output batch dimension comes from k (the query features being attended to)
    assert out.shape == (16, 5, 512), f"Expected shape (16, 5, 512), got {out.shape}"
    print("✓ Cosine attention successful")


def test_forward_pass():
    """Test full forward pass"""
    print("\n=== Testing Full Forward Pass ===")
    
    # Create dummy model
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16
    ).to(device)
    
    # Create dummy input (N-way, K-shot + Q-query, C, H, W)
    x = torch.randn(5, 21, 3, 84, 84)  # 5 + 16 = 21 samples per class
    
    # Forward pass
    scores = model.set_forward(x, is_feature=False)
    
    print(f"Output scores shape: {scores.shape}")
    assert scores.shape == (80, 5), f"Expected shape (80, 5), got {scores.shape}"
    print("✓ Forward pass successful")


def test_loss_computation():
    """Test loss computation with VIC regularization"""
    print("\n=== Testing Loss Computation ===")
    
    # Create dummy model
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16
    ).to(device)
    
    # Create dummy input
    x = torch.randn(5, 21, 3, 84, 84)
    
    # Compute loss
    acc, loss = model.set_forward_loss(x)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Loss: {loss.item():.4f}")
    
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
    assert loss.item() > 0, "Loss should be positive"
    print("✓ Loss computation successful")


def test_dynamic_vic_weights():
    """Test dynamic VIC weight adaptation"""
    print("\n=== Testing Dynamic VIC Weights ===")
    
    # Create dummy model
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16,
        lambda_vic=0.1
    ).to(device)
    
    # Test with low hardness (samples close to prototypes)
    Z_S_easy = torch.randn(5, 5, 512) * 0.1  # Small variance
    h_bar_easy, _ = model.compute_hardness_scores(Z_S_easy)
    
    # Test with high hardness (samples far from prototypes)
    Z_S_hard = torch.randn(5, 5, 512) * 2.0  # Large variance
    h_bar_hard, _ = model.compute_hardness_scores(Z_S_hard)
    
    print(f"Easy hardness: {h_bar_easy.item():.4f}")
    print(f"Hard hardness: {h_bar_hard.item():.4f}")
    
    # Compute dynamic weights
    alpha_V_easy = 0.5 + 0.5 * h_bar_easy.item()
    alpha_V_hard = 0.5 + 0.5 * h_bar_hard.item()
    
    print(f"Easy alpha_V: {alpha_V_easy:.4f}")
    print(f"Hard alpha_V: {alpha_V_hard:.4f}")
    
    assert 0.5 <= alpha_V_easy <= 1.5, "Easy weight should be in reasonable range"
    assert 0.5 <= alpha_V_hard <= 1.5, "Hard weight should be in reasonable range"
    print("✓ Dynamic VIC weight adaptation successful")


def test_gradient_flow():
    """Test that gradients flow through the model"""
    print("\n=== Testing Gradient Flow ===")
    
    # Create dummy model
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16
    ).to(device)
    
    # Create dummy input
    x = torch.randn(5, 21, 3, 84, 84, requires_grad=True)
    
    # Forward and backward
    acc, loss = model.set_forward_loss(x)
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
            break
    
    assert has_grad, "Model should have gradients after backward pass"
    print("✓ Gradient flow successful")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running DV-FSCT Unit Tests")
    print("="*60)
    
    tests = [
        test_vic_variance_loss,
        test_vic_covariance_loss,
        test_hardness_computation,
        test_prototype_computation,
        test_cosine_attention,
        test_forward_pass,
        test_loss_computation,
        test_dynamic_vic_weights,
        test_gradient_flow
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
