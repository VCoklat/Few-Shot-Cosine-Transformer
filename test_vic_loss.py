#!/usr/bin/env python3
"""
Test script to verify VIC loss implementation
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from methods.transformer import FewShotTransformer
import backbone

def test_vic_loss_computation():
    """Test that VIC losses can be computed without errors"""
    print("Testing VIC loss computation...")
    
    # Setup
    n_way = 5
    k_shot = 5
    n_query = 15
    feat_dim = 512
    
    # Create a simple feature model function
    def dummy_feature_model():
        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.final_feat_dim = feat_dim
                self.conv = torch.nn.Linear(3 * 84 * 84, feat_dim)
            
            def forward(self, x):
                batch_size = x.size(0)
                x = x.view(batch_size, -1)
                return self.conv(x)
        
        return DummyBackbone()
    
    # Test 1: Model initialization with default VIC weights (0,0,0)
    print("\nTest 1: Model with default VIC weights (should be backward compatible)")
    model1 = FewShotTransformer(
        dummy_feature_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine'
    )
    assert model1.lambda_I == 1.0, "Default lambda_I should be 1.0"
    assert model1.lambda_V == 0.0, "Default lambda_V should be 0.0"
    assert model1.lambda_C == 0.0, "Default lambda_C should be 0.0"
    print("✓ Default weights are correct")
    
    # Test 2: Model initialization with custom VIC weights
    print("\nTest 2: Model with custom VIC weights")
    model2 = FewShotTransformer(
        dummy_feature_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine',
        lambda_I=1.0,
        lambda_V=0.5,
        lambda_C=0.1
    )
    assert model2.lambda_I == 1.0
    assert model2.lambda_V == 0.5
    assert model2.lambda_C == 0.1
    print("✓ Custom weights set correctly")
    
    # Test 3: Variance loss computation
    print("\nTest 3: Variance loss computation")
    z_support = torch.randn(n_way, k_shot, feat_dim)
    loss_v = model2.variance_loss(z_support)
    assert loss_v.dim() == 0, "Variance loss should be a scalar"
    assert loss_v >= 0, "Variance loss should be non-negative"
    print(f"✓ Variance loss computed: {loss_v.item():.6f}")
    
    # Test 4: Covariance loss computation
    print("\nTest 4: Covariance loss computation")
    loss_c = model2.covariance_loss(z_support)
    assert loss_c.dim() == 0, "Covariance loss should be a scalar"
    assert loss_c >= 0, "Covariance loss should be non-negative"
    print(f"✓ Covariance loss computed: {loss_c.item():.6f}")
    
    # Test 5: Forward pass with combined loss
    print("\nTest 5: Forward pass with combined loss")
    device = torch.device('cpu')
    model2 = model2.to(device)
    
    # Create dummy input
    batch_size = n_way * (k_shot + n_query)
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
    
    acc, loss_total = model2.set_forward_loss(x)
    
    assert isinstance(acc, float), "Accuracy should be a float"
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
    assert loss_total.dim() == 0, "Total loss should be a scalar"
    assert loss_total.requires_grad, "Total loss should require grad"
    print(f"✓ Forward pass successful - Acc: {acc:.4f}, Loss: {loss_total.item():.6f}")
    
    # Test 6: Backward pass
    print("\nTest 6: Backward pass")
    loss_total.backward()
    
    # Check that gradients were computed
    has_grad = False
    for param in model2.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Gradients should be computed"
    print("✓ Backward pass successful, gradients computed")
    
    # Test 7: Model with only variance loss
    print("\nTest 7: Model with only variance loss (lambda_C=0)")
    model3 = FewShotTransformer(
        dummy_feature_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine',
        lambda_I=1.0,
        lambda_V=0.5,
        lambda_C=0.0
    )
    model3 = model3.to(device)
    acc, loss = model3.set_forward_loss(x)
    print(f"✓ Model with V loss only - Acc: {acc:.4f}, Loss: {loss.item():.6f}")
    
    # Test 8: Model with only covariance loss
    print("\nTest 8: Model with only covariance loss (lambda_V=0)")
    model4 = FewShotTransformer(
        dummy_feature_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine',
        lambda_I=1.0,
        lambda_V=0.0,
        lambda_C=0.1
    )
    model4 = model4.to(device)
    acc, loss = model4.set_forward_loss(x)
    print(f"✓ Model with C loss only - Acc: {acc:.4f}, Loss: {loss.item():.6f}")
    
    print("\n" + "="*50)
    print("✓ ALL TESTS PASSED!")
    print("="*50)
    
    return True

if __name__ == '__main__':
    try:
        success = test_vic_loss_computation()
        if success:
            sys.exit(0)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
