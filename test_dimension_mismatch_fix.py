#!/usr/bin/env python3
"""
Test to validate the sequence dimension mismatch fix.
This test ensures that cross-attention works correctly with different batch sizes
and no dimension mismatch warnings appear during training.
"""

import torch
import numpy as np
from methods.transformer import FewShotTransformer, Attention
from backbone import Conv4
import io
import contextlib


def test_no_dimension_mismatch_warnings():
    """Test that no dimension mismatch warnings appear during training"""
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def model_func():
        return Conv4(dataset='miniImagenet')
    
    model = FewShotTransformer(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant="cosine",
        depth=1,
        heads=8,
        dim_head=64
    )
    
    dummy_input = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    model.train()
    model.to('cpu')
    dummy_input = dummy_input.to('cpu')
    
    # Capture output to check for warnings
    output = io.StringIO()
    
    with contextlib.redirect_stdout(output):
        for i in range(5):
            acc, loss = model.set_forward_loss(dummy_input)
    
    output_text = output.getvalue()
    
    # Check for warning messages
    assert "Warning: Sequence dimension mismatch" not in output_text, \
        "Dimension mismatch warning found in output"
    
    print("✓ Test passed: No dimension mismatch warnings")


def test_cross_attention_with_different_batch_sizes():
    """Test that cross-attention handles different batch sizes correctly"""
    # Simulate the shapes in cross-attention
    heads = 8
    batch_q = 1  # prototypes
    batch_k = 75  # queries (5-way * 15-query)
    seq_q = 5
    seq_k = 1
    dim = 64
    
    f_q = torch.randn(heads, batch_q, seq_q, dim)
    f_k = torch.randn(heads, batch_k, seq_k, dim)
    f_v = torch.randn(heads, batch_k, seq_k, dim)
    
    # Test matrix multiplication compatibility
    dots = torch.matmul(f_q, f_k.transpose(-1, -2))
    out = torch.matmul(dots, f_v)
    
    # Verify shapes - broadcasting should work
    expected_out_shape = (heads, batch_k, seq_q, dim)
    assert out.shape == expected_out_shape, \
        f"Output shape {out.shape} != expected {expected_out_shape}"
    
    print("✓ Test passed: Cross-attention with different batch sizes works correctly")


def test_attention_components_cross_attention():
    """Test that attention components handle cross-attention properly"""
    # Create attention module
    dim = 512
    heads = 8
    dim_head = 64
    
    attn = Attention(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        variant="cosine",
        n_way=5,
        k_shot=5
    )
    
    # Simulate cross-attention inputs
    batch_q = 1
    batch_k = 75
    seq_q = 5
    seq_k = 1
    
    f_q = torch.randn(heads, batch_q, seq_q, dim_head)
    f_k = torch.randn(heads, batch_k, seq_k, dim_head)
    
    # Test basic attention components
    cov_basic, var_basic = attn.basic_attention_components(f_q, f_k)
    
    # Both should have shape [heads, batch_k, seq_q, seq_k] due to broadcasting
    expected_shape = (heads, batch_k, seq_q, seq_k)
    assert cov_basic.shape == expected_shape, \
        f"Basic cov shape {cov_basic.shape} != expected {expected_shape}"
    assert var_basic.shape == expected_shape, \
        f"Basic var shape {var_basic.shape} != expected {expected_shape}"
    
    # Test advanced attention components (should fall back to basic for cross-attention)
    cov_adv, var_adv = attn.advanced_attention_components(f_q, f_k, gamma=0.1, epsilon=1e-8)
    
    assert cov_adv.shape == expected_shape, \
        f"Advanced cov shape {cov_adv.shape} != expected {expected_shape}"
    assert var_adv.shape == expected_shape, \
        f"Advanced var shape {var_adv.shape} != expected {expected_shape}"
    
    print("✓ Test passed: Attention components handle cross-attention correctly")


def test_forward_pass_both_attention_modes():
    """Test complete forward pass with both basic and advanced attention"""
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def model_func():
        return Conv4(dataset='miniImagenet')
    
    model = FewShotTransformer(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant="cosine",
        depth=1,
        heads=8,
        dim_head=64
    )
    
    dummy_input = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    model.train()
    model.to('cpu')
    dummy_input = dummy_input.to('cpu')
    
    # Test with advanced attention
    model.use_advanced_attention = True
    acc, loss = model.set_forward_loss(dummy_input)
    assert acc >= 0 and acc <= 1, f"Invalid accuracy: {acc}"
    assert loss > 0, f"Invalid loss: {loss}"
    
    # Test with basic attention
    model.use_advanced_attention = False
    acc, loss = model.set_forward_loss(dummy_input)
    assert acc >= 0 and acc <= 1, f"Invalid accuracy: {acc}"
    assert loss > 0, f"Invalid loss: {loss}"
    
    print("✓ Test passed: Forward pass works with both attention modes")


if __name__ == "__main__":
    print("Running sequence dimension mismatch fix tests...\n")
    
    test_no_dimension_mismatch_warnings()
    test_cross_attention_with_different_batch_sizes()
    test_attention_components_cross_attention()
    test_forward_pass_both_attention_modes()
    
    print("\n✓ All tests passed!")
