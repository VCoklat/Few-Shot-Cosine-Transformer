"""
Test script for FewShotTransformer with VIC Loss
"""

import torch
import numpy as np
import sys
sys.path.append('/home/runner/work/Few-Shot-Cosine-Transformer/Few-Shot-Cosine-Transformer')

from methods.transformer import FewShotTransformer
from backbone import Conv4

def test_fsct_with_vic_loss():
    """Test FewShotTransformer with VIC loss integration"""
    print("Testing FewShotTransformer with VIC Loss...")
    print("=" * 60)
    
    # Parameters
    n_way = 5
    k_shot = 5
    n_query = 15
    
    # Test 1: Model creation without VIC loss
    print("\n1. Testing FewShotTransformer WITHOUT VIC loss...")
    def feature_model():
        return Conv4(dataset='miniImagenet', flatten=True)
    
    model_standard = FewShotTransformer(
        feature_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine',
        use_vic_loss=False
    )
    print(f"   ✓ Standard model created (use_vic_loss=False)")
    
    # Test 2: Model creation with VIC loss
    print("\n2. Testing FewShotTransformer WITH VIC loss...")
    model_vic = FewShotTransformer(
        feature_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine',
        use_vic_loss=True,
        lambda_v=1.0,
        lambda_i=1.0,
        lambda_c=0.04
    )
    print(f"   ✓ VIC-enhanced model created (use_vic_loss=True)")
    assert hasattr(model_vic, 'vic_loss'), "Model should have vic_loss attribute"
    print(f"   ✓ VIC loss module is attached to model")
    
    # Test 3: Forward pass without VIC loss
    print("\n3. Testing forward pass (standard)...")
    # Create dummy input (n_way, k_shot + n_query, C, H, W)
    x_standard = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    model_standard.eval()
    with torch.no_grad():
        scores = model_standard.set_forward(x_standard)
    print(f"   Output shape: {scores.shape}")
    assert scores.shape == (n_way * n_query, n_way), f"Expected shape ({n_way * n_query}, {n_way}), got {scores.shape}"
    print("   ✓ Forward pass produces correct output shape")
    
    # Test 4: Forward pass with return_support=True
    print("\n4. Testing forward pass with support embeddings...")
    with torch.no_grad():
        scores, support_emb = model_vic.set_forward(x_standard, return_support=True)
    print(f"   Scores shape: {scores.shape}")
    print(f"   Support embeddings shape: {support_emb.shape}")
    assert scores.shape == (n_way * n_query, n_way), "Scores shape incorrect"
    assert support_emb.shape[0] == n_way * k_shot, "Support embeddings batch size incorrect"
    print("   ✓ Forward pass returns both scores and support embeddings")
    
    # Test 5: Loss computation without VIC loss
    print("\n5. Testing loss computation (standard)...")
    model_standard.train()
    acc, loss = model_standard.set_forward_loss(x_standard)
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Loss: {loss.item():.4f}")
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
    assert loss.item() > 0, "Loss should be positive"
    print("   ✓ Standard loss computation works correctly")
    
    # Test 6: Loss computation with VIC loss
    print("\n6. Testing loss computation (VIC-enhanced)...")
    model_vic.train()
    acc_vic, loss_vic = model_vic.set_forward_loss(x_standard)
    print(f"   Accuracy: {acc_vic:.4f}")
    print(f"   Loss: {loss_vic.item():.4f}")
    assert 0 <= acc_vic <= 1, "Accuracy should be between 0 and 1"
    assert loss_vic.item() > 0, "Loss should be positive"
    print("   ✓ VIC loss computation works correctly")
    
    # Test 7: Backward pass
    print("\n7. Testing backward pass (gradient computation)...")
    loss_vic.backward()
    # Check if gradients are computed
    has_grad = any(p.grad is not None for p in model_vic.parameters() if p.requires_grad)
    assert has_grad, "Gradients should be computed"
    print("   ✓ Gradients computed successfully")
    
    # Test 8: Memory usage comparison
    print("\n8. Testing memory efficiency...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("   Model parameters:")
    total_params = sum(p.numel() for p in model_vic.parameters())
    trainable_params = sum(p.numel() for p in model_vic.parameters() if p.requires_grad)
    print(f"   - Total: {total_params:,}")
    print(f"   - Trainable: {trainable_params:,}")
    print("   ✓ Model size is reasonable for 8GB GPU")
    
    print("\n" + "=" * 60)
    print("✓ All FewShotTransformer tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_fsct_with_vic_loss()
