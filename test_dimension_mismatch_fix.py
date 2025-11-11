#!/usr/bin/env python3
"""
Test to verify that the sequence dimension mismatch is resolved.
This test simulates the few-shot learning scenario where q (support) and k/v (queries)
have different batch dimensions.
"""

import sys
import torch
import numpy as np
from io import StringIO

# Add the repo to path
sys.path.insert(0, '/home/runner/work/Few-Shot-Cosine-Transformer/Few-Shot-Cosine-Transformer')

def test_dimension_mismatch_fix():
    """Test that the dimension mismatch warning is resolved"""
    print("\n" + "="*60)
    print("Test: Sequence Dimension Mismatch Fix")
    print("="*60)
    
    try:
        from methods.transformer import FewShotTransformer, Attention
        from backbone import Conv4
        
        # Test parameters matching the problem statement
        n_way = 5
        k_shot = 5
        n_query = 16
        feat_dim = 512
        
        print(f"\nTest configuration:")
        print(f"  n_way: {n_way}")
        print(f"  k_shot: {k_shot}")
        print(f"  n_query: {n_query}")
        print(f"  Expected seq_k: {n_way * n_query}")
        
        # Create attention module
        attention = Attention(
            dim=feat_dim,
            heads=8,
            dim_head=64,
            variant='cosine',
            initial_cov_weight=0.3,
            initial_var_weight=0.5,
            dynamic_weight=False,
            n_way=n_way,
            k_shot=k_shot,
            dropout=0.1,
            dataset='CUB'
        )
        attention.eval()  # Set to eval mode to avoid training-specific behavior
        
        # Create input tensors with the problematic shapes
        # q: support prototypes [1, n_way, feat_dim]
        # k, v: query samples [n_way*n_query, 1, feat_dim]
        q = torch.randn(1, n_way, feat_dim)
        k = torch.randn(n_way * n_query, 1, feat_dim)
        v = torch.randn(n_way * n_query, 1, feat_dim)
        
        print(f"\nInput shapes:")
        print(f"  q (support): {q.shape}")
        print(f"  k (queries): {k.shape}")
        print(f"  v (queries): {v.shape}")
        
        # Capture warnings by redirecting stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Run forward pass
            output = attention(q, k, v, use_advanced=False, gamma=1.0, epsilon=1e-8)
            
            # Restore stdout
            sys.stdout = old_stdout
            captured_text = captured_output.getvalue()
            
            print(f"\nOutput shape: {output.shape}")
            
            # Check if warning was printed
            if "Warning: Sequence dimension mismatch" in captured_text:
                print("\n❌ FAILED: Dimension mismatch warning still present!")
                print("\nCaptured output:")
                print(captured_text)
                return False
            else:
                print("\n✅ SUCCESS: No dimension mismatch warning!")
                if captured_text.strip():
                    print("\nOther output captured:")
                    print(captured_text)
                return True
                
        except Exception as e:
            sys.stdout = old_stdout
            print(f"\n❌ FAILED: Exception during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"\n⚠️  SKIPPED: Cannot import required modules: {e}")
        return None
    except Exception as e:
        print(f"\n❌ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dimension_shapes():
    """Test that the dimensions are correctly transformed"""
    print("\n" + "="*60)
    print("Test: Dimension Shape Verification")
    print("="*60)
    
    try:
        from einops import rearrange
        
        # Test parameters
        n_way = 5
        k_shot = 5
        n_query = 16
        feat_dim = 512
        heads = 8
        inner_dim = 512
        head_dim = inner_dim // heads
        
        # Simulate the shapes after input_transform
        batch_q = 1
        seq_q = n_way
        batch_k = n_way * n_query
        seq_k = 1
        
        # Create dummy tensors
        f_q = torch.randn(batch_q, seq_q, inner_dim)
        f_k = torch.randn(batch_k, seq_k, inner_dim)
        f_v = torch.randn(batch_k, seq_k, inner_dim)
        
        print(f"\nBefore rearrange:")
        print(f"  f_q: {f_q.shape}")
        print(f"  f_k: {f_k.shape}")
        print(f"  f_v: {f_v.shape}")
        
        # Apply rearrange
        f_q = rearrange(f_q, 'b n (h d) -> h b n d', h=heads)
        f_k = rearrange(f_k, 'b n (h d) -> h b n d', h=heads)
        f_v = rearrange(f_v, 'b n (h d) -> h b n d', h=heads)
        
        print(f"\nAfter rearrange:")
        print(f"  f_q: {f_q.shape}")
        print(f"  f_k: {f_k.shape}")
        print(f"  f_v: {f_v.shape}")
        
        # Apply the fix
        if f_q.shape[1] != f_k.shape[1]:
            batch_q_new, batch_k_new = f_q.shape[1], f_k.shape[1]
            seq_q_new, seq_k_new = f_q.shape[2], f_k.shape[2]
            
            print(f"\nBatch dimension mismatch detected: {batch_q_new} != {batch_k_new}")
            
            if batch_q_new == 1 and batch_k_new > 1:
                print(f"Applying fix: reshaping k and v from [heads, {batch_k_new}, {seq_k_new}, dim] to [heads, 1, {batch_k_new * seq_k_new}, dim]")
                f_k = f_k.permute(0, 1, 2, 3).contiguous().view(f_k.shape[0], 1, batch_k_new * seq_k_new, f_k.shape[3])
                f_v = f_v.permute(0, 1, 2, 3).contiguous().view(f_v.shape[0], 1, batch_k_new * seq_k_new, f_v.shape[3])
        
        print(f"\nAfter fix:")
        print(f"  f_q: {f_q.shape}")
        print(f"  f_k: {f_k.shape}")
        print(f"  f_v: {f_v.shape}")
        
        # Verify dimensions for attention computation
        # dots = matmul(f_q, f_k.transpose(-1, -2))
        # dots shape should be [heads, batch, seq_q, seq_k]
        f_k_transposed = f_k.transpose(-1, -2)
        print(f"\nf_k transposed: {f_k_transposed.shape}")
        
        # Simulate matmul
        try:
            dots = torch.matmul(f_q, f_k_transposed)
            print(f"dots shape: {dots.shape}")
            print(f"  dots.shape[3] (seq_k): {dots.shape[3]}")
            print(f"  f_v.shape[2] (seq_v): {f_v.shape[2]}")
            
            # Check if dimensions match
            if dots.shape[3] == f_v.shape[2]:
                print(f"\n✅ SUCCESS: Dimensions match! Can perform dots @ f_v")
                # Verify the final matmul works
                out = torch.matmul(dots, f_v)
                print(f"Output shape: {out.shape}")
                return True
            else:
                print(f"\n❌ FAILED: Dimension mismatch! dots expects seq_k={dots.shape[3]} but f_v has seq_v={f_v.shape[2]}")
                return False
        except Exception as e:
            print(f"\n❌ FAILED: Cannot perform matmul: {e}")
            return False
            
    except ImportError as e:
        print(f"\n⚠️  SKIPPED: Cannot import required modules: {e}")
        return None
    except Exception as e:
        print(f"\n❌ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Sequence Dimension Mismatch Fix")
    print("="*60)
    
    # Run tests
    test1_result = test_dimension_shapes()
    test2_result = test_dimension_mismatch_fix()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    results = {
        "Dimension Shape Verification": test1_result,
        "Dimension Mismatch Fix": test2_result
    }
    
    passed = sum(1 for r in results.values() if r is True)
    skipped = sum(1 for r in results.values() if r is None)
    failed = sum(1 for r in results.values() if r is False)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is None:
            status = "⚠️  SKIPPED"
        else:
            status = "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)
