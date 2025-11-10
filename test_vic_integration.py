"""
Test script to verify VIC regularization integration works correctly.
This is a minimal test that doesn't require dataset or extensive training.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.vic_regularization import VICRegularization
from methods.transformer import FewShotTransformer
import backbone

def test_vic_regularization_basic():
    """Test basic VIC regularization functionality."""
    print("=" * 60)
    print("Test 1: VIC Regularization Basic Functionality")
    print("=" * 60)
    
    # Create VIC regularization module
    vic = VICRegularization(
        lambda_v=1.0,
        lambda_i=1.0,
        lambda_c=1.0,
        epsilon=1e-4,
        alpha=0.001
    )
    
    # Create dummy embeddings (5-way, 5-shot, 512-dim)
    n_way = 5
    k_shot = 5
    dim = 512
    support_embeddings = torch.randn(n_way, k_shot, dim)
    query_embeddings = torch.randn(n_way * 15, dim)  # 15 queries per class
    
    print(f"Support embeddings shape: {support_embeddings.shape}")
    print(f"Query embeddings shape: {query_embeddings.shape}")
    
    # Test variance loss
    v_loss = vic.variance_loss(support_embeddings)
    print(f"\nVariance loss: {v_loss.item():.6f}")
    assert v_loss.item() >= 0, "Variance loss should be non-negative"
    
    # Test invariance loss
    i_loss = vic.invariance_loss(support_embeddings)
    print(f"Invariance loss: {i_loss.item():.6f}")
    assert i_loss.item() >= 0, "Invariance loss should be non-negative"
    
    # Test covariance loss
    c_loss = vic.covariance_loss(support_embeddings)
    print(f"Covariance loss: {c_loss.item():.6f}")
    assert c_loss.item() >= 0, "Covariance loss should be non-negative"
    
    # Test full forward pass
    vic.train()
    vic_dict = vic(support_embeddings, query_embeddings)
    print(f"\nTotal VIC loss: {vic_dict['total'].item():.6f}")
    print(f"Lambda_v: {vic_dict['lambda_v'].item():.4f}")
    print(f"Lambda_i: {vic_dict['lambda_i'].item():.4f}")
    print(f"Lambda_c: {vic_dict['lambda_c'].item():.4f}")
    
    # Test dynamic weight updates
    vic.update_dynamic_weights(
        vic_dict['variance'].detach(),
        vic_dict['invariance'].detach(),
        vic_dict['covariance'].detach()
    )
    
    stats = vic.get_weight_stats()
    print(f"\nWeight stats after update:")
    print(f"  Lambda_v: {stats['lambda_v']:.4f}")
    print(f"  Lambda_i: {stats['lambda_i']:.4f}")
    print(f"  Lambda_c: {stats['lambda_c']:.4f}")
    
    print("\n✅ Test 1 passed!")
    return True


def test_transformer_integration():
    """Test VIC regularization integration with FewShotTransformer."""
    print("\n" + "=" * 60)
    print("Test 2: Transformer Integration")
    print("=" * 60)
    
    # Setup
    n_way = 5
    k_shot = 5
    n_query = 15
    
    # Create a simple feature model
    def feature_model():
        model = backbone.Conv4('miniImagenet', flatten=True)
        return model
    
    # Test without VIC
    print("\nCreating model WITHOUT VIC regularization...")
    model_no_vic = FewShotTransformer(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        use_vic=False
    )
    print(f"Model created successfully")
    print(f"Feature dimension: {model_no_vic.feat_dim}")
    
    # Test with VIC
    print("\nCreating model WITH VIC regularization...")
    model_with_vic = FewShotTransformer(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        use_vic=True,
        vic_lambda_v=1.0,
        vic_lambda_i=1.0,
        vic_lambda_c=1.0
    )
    print(f"Model created successfully")
    print(f"VIC regularization enabled: {model_with_vic.use_vic}")
    
    # Create dummy input (n_way, k_shot + n_query, C, H, W)
    # For Conv4 with miniImagenet, input is 84x84x3
    batch_size = n_way
    total_samples = k_shot + n_query
    x = torch.randn(batch_size, total_samples, 3, 84, 84)
    
    print(f"\nInput shape: {x.shape}")
    
    # Test forward pass without VIC
    print("\nTesting forward pass without VIC...")
    model_no_vic.eval()
    with torch.no_grad():
        scores_no_vic = model_no_vic.set_forward(x)
    print(f"Output shape (no VIC): {scores_no_vic.shape}")
    assert scores_no_vic.shape == (n_way * n_query, n_way), f"Expected shape {(n_way * n_query, n_way)}, got {scores_no_vic.shape}"
    
    # Test forward pass with VIC
    print("\nTesting forward pass with VIC...")
    model_with_vic.eval()
    with torch.no_grad():
        scores_with_vic = model_with_vic.set_forward(x)
    print(f"Output shape (with VIC): {scores_with_vic.shape}")
    assert scores_with_vic.shape == (n_way * n_query, n_way), f"Expected shape {(n_way * n_query, n_way)}, got {scores_with_vic.shape}"
    
    # Test training mode with loss
    print("\nTesting training mode with VIC loss...")
    model_with_vic.train()
    result = model_with_vic.set_forward_loss(x)
    
    if len(result) == 3:
        acc, loss, vic_dict = result
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"Total loss: {loss.item():.6f}")
        print(f"VIC variance loss: {vic_dict['variance'].item():.6f}")
        print(f"VIC invariance loss: {vic_dict['invariance'].item():.6f}")
        print(f"VIC covariance loss: {vic_dict['covariance'].item():.6f}")
    else:
        print("❌ Expected 3 return values (acc, loss, vic_dict)")
        return False
    
    print("\n✅ Test 2 passed!")
    return True


def test_backward_pass():
    """Test that gradients flow correctly through VIC regularization."""
    print("\n" + "=" * 60)
    print("Test 3: Backward Pass with VIC")
    print("=" * 60)
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def feature_model():
        return backbone.Conv4('miniImagenet', flatten=True)
    
    model = FewShotTransformer(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        use_vic=True,
        vic_lambda_v=1.0,
        vic_lambda_i=1.0,
        vic_lambda_c=1.0
    )
    
    # Create dummy input
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Forward pass
    model.train()
    acc, loss, vic_dict = model.set_forward_loss(x)
    
    print(f"Total loss: {loss.item():.6f}")
    
    # Backward pass
    print("\nPerforming backward pass...")
    loss.backward()
    
    # Check that gradients exist
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients found after backward pass"
    print("✅ Gradients computed successfully")
    
    # Check VIC parameters have gradients
    if model.vic_regularization.lambda_v.grad is not None:
        print(f"VIC lambda_v gradient: {model.vic_regularization.lambda_v.grad.item():.6f}")
    
    print("\n✅ Test 3 passed!")
    return True


def test_memory_efficiency():
    """Test memory usage with and without VIC."""
    print("\n" + "=" * 60)
    print("Test 4: Memory Efficiency Check")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return True
    
    device = torch.device('cuda')
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def feature_model():
        return backbone.Conv4('miniImagenet', flatten=True)
    
    # Test without VIC
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model_no_vic = FewShotTransformer(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        use_vic=False
    ).to(device)
    
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
    
    model_no_vic.train()
    acc, loss = model_no_vic.set_forward_loss(x)
    loss.backward()
    
    mem_no_vic = torch.cuda.max_memory_allocated() / 1024**2  # MB
    print(f"Peak memory without VIC: {mem_no_vic:.2f} MB")
    
    # Test with VIC
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model_with_vic = FewShotTransformer(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        use_vic=True
    ).to(device)
    
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
    
    model_with_vic.train()
    acc, loss, vic_dict = model_with_vic.set_forward_loss(x)
    loss.backward()
    
    mem_with_vic = torch.cuda.max_memory_allocated() / 1024**2  # MB
    print(f"Peak memory with VIC: {mem_with_vic:.2f} MB")
    
    mem_increase = mem_with_vic - mem_no_vic
    mem_increase_pct = (mem_increase / mem_no_vic) * 100
    print(f"Memory increase: {mem_increase:.2f} MB ({mem_increase_pct:.1f}%)")
    
    # Check that memory increase is reasonable (should be < 50% increase)
    assert mem_increase_pct < 50, f"Memory increase too high: {mem_increase_pct:.1f}%"
    
    print("\n✅ Test 4 passed!")
    return True


if __name__ == "__main__":
    print("Testing VIC Regularization Integration")
    print("=" * 60)
    
    try:
        # Run all tests
        all_passed = True
        
        all_passed &= test_vic_regularization_basic()
        all_passed &= test_transformer_integration()
        all_passed &= test_backward_pass()
        all_passed &= test_memory_efficiency()
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
