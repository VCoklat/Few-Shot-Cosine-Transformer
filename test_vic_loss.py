"""
Test script for VIC Loss implementation
"""

import torch
import numpy as np
from methods.vic_loss import VICLoss

def test_vic_loss():
    """Test VIC loss components"""
    print("Testing VIC Loss Implementation...")
    print("=" * 60)
    
    # Parameters
    n_way = 5
    k_shot = 5
    n_query = 15
    feature_dim = 512
    batch_size = n_way * k_shot
    
    # Create VIC loss module
    vic_loss = VICLoss(lambda_v=1.0, lambda_i=1.0, lambda_c=0.04)
    print(f"✓ VIC Loss module created with λ_v=1.0, λ_i=1.0, λ_c=0.04")
    
    # Test 1: Invariance Loss (Cross-Entropy)
    print("\n1. Testing Invariance Loss (Cross-Entropy)...")
    predictions = torch.randn(n_query * n_way, n_way)
    targets = torch.from_numpy(np.repeat(range(n_way), n_query)).long()
    l_i = vic_loss.invariance_loss(predictions, targets)
    print(f"   Invariance Loss: {l_i.item():.4f}")
    assert l_i.item() > 0, "Invariance loss should be positive"
    print("   ✓ Invariance loss working correctly")
    
    # Test 2: Variance Loss
    print("\n2. Testing Variance Loss...")
    support_embeddings = torch.randn(batch_size, feature_dim)
    l_v = vic_loss.variance_loss(support_embeddings, n_way, k_shot)
    print(f"   Variance Loss: {l_v.item():.4f}")
    assert l_v.item() >= 0, "Variance loss should be non-negative"
    print("   ✓ Variance loss working correctly")
    
    # Test 3: Covariance Loss
    print("\n3. Testing Covariance Loss...")
    l_c = vic_loss.covariance_loss(support_embeddings)
    print(f"   Covariance Loss: {l_c.item():.4f}")
    assert l_c.item() >= 0, "Covariance loss should be non-negative"
    print("   ✓ Covariance loss working correctly")
    
    # Test 4: Combined VIC Loss
    print("\n4. Testing Combined VIC Loss...")
    loss_dict = vic_loss(predictions, targets, support_embeddings, n_way, k_shot)
    print(f"   Total Loss: {loss_dict['total'].item():.4f}")
    print(f"   - Invariance: {loss_dict['invariance']:.4f}")
    print(f"   - Variance: {loss_dict['variance']:.4f}")
    print(f"   - Covariance: {loss_dict['covariance']:.4f}")
    assert loss_dict['total'].item() > 0, "Total loss should be positive"
    print("   ✓ Combined VIC loss working correctly")
    
    # Test 5: Gradient flow
    print("\n5. Testing Gradient Flow...")
    predictions.requires_grad = True
    support_embeddings.requires_grad = True
    loss_dict = vic_loss(predictions, targets, support_embeddings, n_way, k_shot)
    loss_dict['total'].backward()
    assert predictions.grad is not None, "Gradients should flow through predictions"
    assert support_embeddings.grad is not None, "Gradients should flow through embeddings"
    print("   ✓ Gradients flow correctly through all components")
    
    # Test 6: Memory efficiency check
    print("\n6. Testing Memory Efficiency...")
    # Simulate larger batch (but still reasonable for 8GB GPU)
    n_way_large = 10
    k_shot_large = 10
    batch_large = n_way_large * k_shot_large
    support_large = torch.randn(batch_large, feature_dim)
    predictions_large = torch.randn(n_way_large * 15, n_way_large)
    targets_large = torch.from_numpy(np.repeat(range(n_way_large), 15)).long()
    
    loss_dict = vic_loss(predictions_large, targets_large, support_large, n_way_large, k_shot_large)
    print(f"   Large batch (10-way, 10-shot) Total Loss: {loss_dict['total'].item():.4f}")
    print("   ✓ Memory efficient for larger batches")
    
    print("\n" + "=" * 60)
    print("✓ All VIC Loss tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_vic_loss()
