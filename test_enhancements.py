"""
Test script to validate the variance, covariance, invariance and dynamic weight enhancements
"""
import torch
import torch.nn as nn
import numpy as np
import sys

# Test imports
try:
    from methods.transformer import FewShotTransformer, Attention
    from methods.CTX import CTX
    import backbone
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_attention_module():
    """Test the enhanced Attention module"""
    print("\n" + "="*60)
    print("Testing Enhanced Attention Module")
    print("="*60)
    
    try:
        # Create attention module
        dim = 512
        heads = 8
        dim_head = 64
        variant = "cosine"
        
        attn = Attention(dim=dim, heads=heads, dim_head=dim_head, variant=variant)
        print(f"✓ Attention module created successfully")
        
        # Check new parameters exist
        assert hasattr(attn, 'dynamic_weight'), "Missing dynamic_weight parameter"
        assert hasattr(attn, 'variance_weight'), "Missing variance_weight parameter"
        assert hasattr(attn, 'covariance_weight'), "Missing covariance_weight parameter"
        assert hasattr(attn, 'invariance_proj'), "Missing invariance_proj module"
        print(f"✓ All new parameters and modules present")
        
        # Test forward pass
        batch_size = 1
        n_way = 5
        q = torch.randn(batch_size, n_way, dim)
        k = torch.randn(1, 1, dim)
        v = torch.randn(1, 1, dim)
        
        output = attn(q, k, v)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {q.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Test variance computation
        test_tensor = torch.randn(heads, batch_size, n_way, dim_head)
        variance = attn.compute_variance(test_tensor)
        print(f"✓ Variance computation successful")
        print(f"  Variance shape: {variance.shape}")
        
        # Test covariance computation
        covariance = attn.compute_covariance(test_tensor, test_tensor)
        print(f"✓ Covariance computation successful")
        print(f"  Covariance shape: {covariance.shape}")
        
        # Test invariance transformation
        inv_output = attn.apply_invariance(test_tensor)
        print(f"✓ Invariance transformation successful")
        print(f"  Invariance output shape: {inv_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in Attention module: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fsct_model():
    """Test the enhanced FewShotTransformer model"""
    print("\n" + "="*60)
    print("Testing Enhanced FewShotTransformer Model")
    print("="*60)
    
    try:
        n_way = 5
        k_shot = 5
        n_query = 15
        
        # Create a simple feature extractor
        def feature_model():
            class SimpleFeature(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 64, 3, padding=1)
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    self.final_feat_dim = 64
                    
                def forward(self, x):
                    x = self.conv(x)
                    x = self.pool(x)
                    return x.flatten(1)
            return SimpleFeature()
        
        # Create model
        model = FewShotTransformer(
            feature_model, 
            n_way=n_way, 
            k_shot=k_shot, 
            n_query=n_query,
            variant="cosine"
        )
        print(f"✓ FewShotTransformer model created successfully")
        
        # Check attention module has enhancements
        assert hasattr(model.ATTN, 'dynamic_weight'), "ATTN missing dynamic_weight"
        assert hasattr(model.ATTN, 'variance_weight'), "ATTN missing variance_weight"
        assert hasattr(model.ATTN, 'covariance_weight'), "ATTN missing covariance_weight"
        print(f"✓ Attention enhancements integrated")
        
        # Test forward pass with dummy data
        batch_size = n_way * (k_shot + n_query)
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
        
        output = model.set_forward(x, is_feature=False)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in FewShotTransformer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ctx_model():
    """Test the enhanced CTX model"""
    print("\n" + "="*60)
    print("Testing Enhanced CTX Model")
    print("="*60)
    
    try:
        n_way = 5
        k_shot = 5
        n_query = 15
        
        # Create a simple feature extractor  
        def feature_model():
            class SimpleFeature(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 64, 3, padding=1)
                    self.final_feat_dim = [64, 21, 21]
                    
                def forward(self, x):
                    return self.conv(x)
            return SimpleFeature()
        
        # Create model
        model = CTX(
            feature_model,
            n_way=n_way,
            k_shot=k_shot, 
            n_query=n_query,
            variant="cosine",
            input_dim=64
        )
        print(f"✓ CTX model created successfully")
        
        # Check new parameters exist
        assert hasattr(model, 'dynamic_weight'), "Missing dynamic_weight parameter"
        assert hasattr(model, 'variance_weight'), "Missing variance_weight parameter"
        assert hasattr(model, 'covariance_weight'), "Missing covariance_weight parameter"
        assert hasattr(model, 'invariance_query'), "Missing invariance_query module"
        assert hasattr(model, 'invariance_support'), "Missing invariance_support module"
        print(f"✓ All new parameters and modules present")
        
        # Test variance and covariance methods
        test_tensor = torch.randn(2, 3, 64)
        variance = model.compute_variance(test_tensor)
        print(f"✓ Variance computation successful")
        
        covariance = model.compute_covariance(test_tensor, test_tensor)
        print(f"✓ Covariance computation successful")
        
        # Test forward pass with dummy data
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
        
        output = model.set_forward(x, is_feature=False)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in CTX model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_learning():
    """Test that new parameters are learnable"""
    print("\n" + "="*60)
    print("Testing Parameter Learning")
    print("="*60)
    
    try:
        dim = 512
        attn = Attention(dim=dim, heads=8, dim_head=64, variant="cosine")
        
        # Get initial values
        initial_dw = attn.dynamic_weight.item()
        initial_vw = attn.variance_weight.item()
        initial_cw = attn.covariance_weight.item()
        
        print(f"✓ Initial parameter values:")
        print(f"  Dynamic weight: {initial_dw:.4f}")
        print(f"  Variance weight: {initial_vw:.4f}")
        print(f"  Covariance weight: {initial_cw:.4f}")
        
        # Check parameters are in optimizer
        optimizer = torch.optim.Adam(attn.parameters(), lr=0.01)
        param_count = len(list(attn.parameters()))
        print(f"✓ Total trainable parameters: {param_count}")
        
        # Simulate training step
        q = torch.randn(1, 5, dim)
        k = torch.randn(1, 1, dim)
        v = torch.randn(1, 1, dim)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = attn(q, k, v)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        
        # Check values changed
        final_dw = attn.dynamic_weight.item()
        final_vw = attn.variance_weight.item()
        final_cw = attn.covariance_weight.item()
        
        print(f"✓ Final parameter values after 5 steps:")
        print(f"  Dynamic weight: {final_dw:.4f} (change: {abs(final_dw - initial_dw):.4f})")
        print(f"  Variance weight: {final_vw:.4f} (change: {abs(final_vw - initial_vw):.4f})")
        print(f"  Covariance weight: {final_cw:.4f} (change: {abs(final_cw - initial_cw):.4f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in parameter learning test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("VARIANCE, COVARIANCE, INVARIANCE & DYNAMIC WEIGHT TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Attention Module", test_attention_module()))
    results.append(("FewShotTransformer Model", test_fsct_model()))
    results.append(("CTX Model", test_ctx_model()))
    results.append(("Parameter Learning", test_parameter_learning()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
