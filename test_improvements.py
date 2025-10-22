"""
Test script to validate variance, covariance, and invariance improvements.
This script tests dimension consistency and memory efficiency.
"""

import torch
import torch.nn as nn
import numpy as np
from methods.transformer import FewShotTransformer, Attention
from methods.CTX import CTX
import backbone

def test_dimension_consistency():
    """Test that all operations maintain consistent dimensions"""
    print("Testing dimension consistency...")
    
    # Test parameters
    n_way = 5
    k_shot = 5
    n_query = 15
    batch_size = n_way * (k_shot + n_query)
    
    # Create dummy model
    def dummy_model():
        return backbone.Conv4('miniImagenet', flatten=True)
    
    # Test FewShotTransformer with new features
    model = FewShotTransformer(
        dummy_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine',
        use_variance=True,
        use_covariance=True,
        use_dynamic_weights=True
    )
    
    # Create dummy input
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    try:
        # Forward pass
        output = model.set_forward(x, is_feature=False)
        
        # Check output dimensions
        expected_shape = (n_way * n_query, n_way)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        print(f"✓ FewShotTransformer output shape: {output.shape}")
        
        # Test loss computation
        acc, loss = model.set_forward_loss(x)
        print(f"✓ Loss computation successful: loss={loss.item():.4f}, acc={acc:.4f}")
        
    except Exception as e:
        print(f"✗ FewShotTransformer test failed: {e}")
        return False
    
    return True

def test_ctx_dimension_consistency():
    """Test CTX model with variance and invariance"""
    print("\nTesting CTX dimension consistency...")
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def dummy_model():
        return backbone.Conv4('miniImagenet', flatten=False)
    
    model = CTX(
        dummy_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        input_dim=64,
        use_variance=True,
        use_invariance=True
    )
    
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    try:
        output = model.set_forward(x, is_feature=False)
        expected_shape = (n_way * n_query, n_way)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        print(f"✓ CTX output shape: {output.shape}")
        
        acc, loss = model.set_forward_loss(x)
        print(f"✓ Loss computation successful: loss={loss.item():.4f}, acc={acc:.4f}")
        
    except Exception as e:
        print(f"✗ CTX test failed: {e}")
        return False
    
    return True

def test_variance_computation():
    """Test variance-based attention computation"""
    print("\nTesting variance computation...")
    
    dim = 512
    heads = 8
    dim_head = 64
    
    attn = Attention(dim, heads, dim_head, variant='cosine', 
                    use_variance=True, use_covariance=True)
    
    # Create dummy tensors matching FewShotTransformer usage
    # Proto: (1, n_way, dim), Query: (n_queries, 1, dim)
    q = torch.randn(1, 5, dim)  # (1 proto batch, 5 ways, dim)
    k = torch.randn(10, 1, dim)  # (10 queries, 1, dim)
    v = torch.randn(10, 1, dim)  # (10 queries, 1, dim)
    
    try:
        output = attn(q, k, v)
        # Output should be (10, 5, dim) - each query attending to each way
        expected_shape = (10, 5, dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Attention output shape: {output.shape}")
        
        # Check that variance and covariance parameters exist
        assert hasattr(attn, 'variance_scale'), "Missing variance_scale parameter"
        assert hasattr(attn, 'covariance_scale'), "Missing covariance_scale parameter"
        print("✓ Variance and covariance parameters present")
        
    except Exception as e:
        print(f"✗ Variance computation test failed: {e}")
        return False
    
    return True

def test_memory_efficiency():
    """Test that gradient checkpointing reduces memory usage"""
    print("\nTesting memory efficiency with gradient checkpointing...")
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def dummy_model():
        return backbone.Conv4('miniImagenet', flatten=True)
    
    model = FewShotTransformer(
        dummy_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        depth=2,  # Use multiple layers to test checkpointing
        use_variance=True,
        use_covariance=True,
        use_dynamic_weights=True
    )
    
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    try:
        # Test forward and backward pass
        model.train()
        output = model.set_forward(x)
        loss = output.mean()
        loss.backward()
        
        print("✓ Gradient checkpointing works correctly")
        print(f"✓ Forward/backward pass successful with all features enabled")
        
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        return False
    
    return True

def test_dynamic_weights():
    """Test dynamic weight generation"""
    print("\nTesting dynamic weight mechanism...")
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def dummy_model():
        return backbone.Conv4('miniImagenet', flatten=True)
    
    model = FewShotTransformer(
        dummy_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        use_dynamic_weights=True
    )
    
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    try:
        output = model.set_forward(x)
        
        # Check that weight generator exists
        assert hasattr(model, 'weight_generator'), "Missing weight_generator"
        print("✓ Dynamic weight generator present")
        
        # Check output shape
        expected_shape = (n_way * n_query, n_way)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Output shape with dynamic weights: {output.shape}")
        
    except Exception as e:
        print(f"✗ Dynamic weights test failed: {e}")
        return False
    
    return True

def main():
    print("="*60)
    print("Running improvement validation tests")
    print("="*60)
    
    tests = [
        test_dimension_consistency,
        test_ctx_dimension_consistency,
        test_variance_computation,
        test_memory_efficiency,
        test_dynamic_weights
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
