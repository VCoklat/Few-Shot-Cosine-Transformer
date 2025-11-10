"""
Unit tests for FSCT_ProFONet hybrid method components

Tests cover:
1. VIC Regularization Module
2. Dynamic Weight Scheduler
3. Cosine Attention Layer
4. Full model integration
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.fsct_profonet import (
    VICRegularization, 
    DynamicWeightScheduler, 
    CosineAttentionLayer,
    FSCT_ProFONet
)
from backbone import Conv4


def test_vic_regularization():
    """Test VIC Regularization Module"""
    print("\n=== Testing VIC Regularization Module ===")
    
    vic_reg = VICRegularization(gamma=1.0, epsilon=1e-6)
    
    # Create sample embeddings
    batch_size = 32
    dim = 128
    embeddings = torch.randn(batch_size, dim)
    
    # Test variance loss
    v_loss = vic_reg.variance_loss(embeddings)
    print(f"Variance loss: {v_loss.item():.6f}")
    assert v_loss.item() >= 0, "Variance loss should be non-negative"
    
    # Test covariance loss
    c_loss = vic_reg.covariance_loss(embeddings)
    print(f"Covariance loss: {c_loss.item():.6f}")
    assert c_loss.item() >= 0, "Covariance loss should be non-negative"
    
    # Test forward pass
    v_loss, c_loss = vic_reg(embeddings)
    print(f"Combined VIC losses - V: {v_loss.item():.6f}, C: {c_loss.item():.6f}")
    
    print("✓ VIC Regularization tests passed")
    return True


def test_dynamic_weight_scheduler():
    """Test Dynamic Weight Scheduler"""
    print("\n=== Testing Dynamic Weight Scheduler ===")
    
    scheduler = DynamicWeightScheduler(
        lambda_V_base=0.5,
        lambda_I=9.0,
        lambda_C_base=0.5
    )
    
    total_epochs = 50
    
    # Test at different epochs
    for epoch in [0, 25, 49]:
        lambda_V, lambda_I, lambda_C = scheduler.get_weights(epoch, total_epochs)
        print(f"Epoch {epoch:2d}/{total_epochs}: λ_V={lambda_V:.4f}, λ_I={lambda_I:.4f}, λ_C={lambda_C:.4f}")
        
        # Check that weights follow expected pattern
        assert lambda_I == 9.0, "Invariance weight should be constant"
        if epoch > 0:
            epoch_ratio = epoch / total_epochs
            expected_V = 0.5 * (1 + 0.3 * epoch_ratio)
            expected_C = 0.5 * (1 - 0.2 * epoch_ratio)
            assert abs(lambda_V - expected_V) < 1e-6, "Variance weight incorrect"
            assert abs(lambda_C - expected_C) < 1e-6, "Covariance weight incorrect"
    
    print("✓ Dynamic Weight Scheduler tests passed")
    return True


def test_cosine_attention_layer():
    """Test Cosine Attention Layer"""
    print("\n=== Testing Cosine Attention Layer ===")
    
    dim = 128
    heads = 4
    dim_head = 32
    
    attn = CosineAttentionLayer(dim=dim, heads=heads, dim_head=dim_head, dropout=0.0)
    
    # Create sample inputs
    batch_q = 15  # n_way * n_query
    n_way = 5
    
    q = torch.randn(batch_q, n_way, dim)  # queries (prototypes)
    k = torch.randn(batch_q, 1, dim)      # keys (query samples)
    v = torch.randn(batch_q, 1, dim)      # values (query samples)
    
    # Forward pass
    output = attn(q, k, v)
    
    print(f"Input shape: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == q.shape, "Output shape should match query shape"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    print("✓ Cosine Attention Layer tests passed")
    return True


def test_fsct_profonet_initialization():
    """Test FSCT_ProFONet model initialization"""
    print("\n=== Testing FSCT_ProFONet Initialization ===")
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def model_func():
        return Conv4(dataset='miniImagenet', flatten=True)
    
    model = FSCT_ProFONet(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        depth=1,
        heads=4,
        dim_head=160,
        mlp_dim=512,
        dropout=0.0,
        lambda_V_base=0.5,
        lambda_I=9.0,
        lambda_C_base=0.5,
        gradient_checkpointing=False,
        mixed_precision=False
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"VIC Regularization: {model.vic_reg}")
    print(f"Weight Scheduler: {model.weight_scheduler}")
    print(f"Feature dimension: {model.feat_dim}")
    
    print("✓ FSCT_ProFONet initialization tests passed")
    return True


def test_fsct_profonet_forward():
    """Test FSCT_ProFONet forward pass"""
    print("\n=== Testing FSCT_ProFONet Forward Pass ===")
    
    n_way = 5
    k_shot = 5
    n_query = 10
    
    def model_func():
        return Conv4(dataset='miniImagenet', flatten=True)
    
    model = FSCT_ProFONet(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        depth=1,
        heads=4,
        dim_head=160,
        mlp_dim=512,
        dropout=0.0,
        gradient_checkpointing=False,
        mixed_precision=False
    )
    
    model.eval()
    
    # Create sample input
    # Shape: (n_way, k_shot + n_query, 3, 84, 84)
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Forward pass
    with torch.no_grad():
        scores, z_support, z_proto = model.set_forward(x, is_feature=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Support embeddings shape: {z_support.shape}")
    print(f"Prototype embeddings shape: {z_proto.shape}")
    print(f"Output scores shape: {scores.shape}")
    
    expected_shape = (n_way * n_query, n_way)
    assert scores.shape == expected_shape, f"Expected scores shape {expected_shape}, got {scores.shape}"
    assert not torch.isnan(scores).any(), "Scores contain NaN values"
    assert not torch.isinf(scores).any(), "Scores contain Inf values"
    
    print("✓ FSCT_ProFONet forward pass tests passed")
    return True


def test_fsct_profonet_loss():
    """Test FSCT_ProFONet loss computation"""
    print("\n=== Testing FSCT_ProFONet Loss Computation ===")
    
    n_way = 5
    k_shot = 5
    n_query = 10
    
    def model_func():
        return Conv4(dataset='miniImagenet', flatten=True)
    
    model = FSCT_ProFONet(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        depth=1,
        heads=4,
        dim_head=160,
        mlp_dim=512,
        dropout=0.0,
        gradient_checkpointing=False,
        mixed_precision=False
    )
    
    model.train()
    
    # Set epoch for dynamic weights
    model.set_epoch(0, 50)
    
    # Create sample input
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Forward pass with loss
    acc, loss = model.set_forward_loss(x)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Loss: {loss.item():.6f}")
    
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss).any(), "Loss contains NaN"
    assert not torch.isinf(loss).any(), "Loss contains Inf"
    
    # Test backpropagation
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
            assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf"
    
    assert has_grad, "No gradients computed"
    
    print("✓ FSCT_ProFONet loss computation tests passed")
    return True


def test_epoch_setting():
    """Test epoch setting for dynamic weights"""
    print("\n=== Testing Epoch Setting ===")
    
    n_way = 5
    k_shot = 5
    n_query = 10
    
    def model_func():
        return Conv4(dataset='miniImagenet', flatten=True)
    
    model = FSCT_ProFONet(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine'
    )
    
    # Test setting different epochs
    model.set_epoch(0, 50)
    lambda_V_0, lambda_I_0, lambda_C_0 = model.weight_scheduler.get_weights(0, 50)
    
    model.set_epoch(25, 50)
    lambda_V_25, lambda_I_25, lambda_C_25 = model.weight_scheduler.get_weights(25, 50)
    
    print(f"Epoch  0: λ_V={lambda_V_0:.4f}, λ_I={lambda_I_0:.4f}, λ_C={lambda_C_0:.4f}")
    print(f"Epoch 25: λ_V={lambda_V_25:.4f}, λ_I={lambda_I_25:.4f}, λ_C={lambda_C_25:.4f}")
    
    # Verify expected changes
    assert lambda_V_25 > lambda_V_0, "Variance weight should increase over epochs"
    assert lambda_C_25 < lambda_C_0, "Covariance weight should decrease over epochs"
    assert lambda_I_25 == lambda_I_0, "Invariance weight should remain constant"
    
    print("✓ Epoch setting tests passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running FSCT_ProFONet Unit Tests")
    print("="*60)
    
    tests = [
        test_vic_regularization,
        test_dynamic_weight_scheduler,
        test_cosine_attention_layer,
        test_fsct_profonet_initialization,
        test_fsct_profonet_forward,
        test_fsct_profonet_loss,
        test_epoch_setting
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED with error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
