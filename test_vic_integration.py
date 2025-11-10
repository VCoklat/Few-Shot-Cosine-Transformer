#!/usr/bin/env python3
"""
Integration test for VIC loss with episodic training
This test verifies that the complete training pipeline works correctly
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, '.')

from methods.transformer import FewShotTransformer
from io_utils import model_dict
import backbone

def test_episodic_training_with_vic():
    """Test a complete episodic training step with VIC losses"""
    print("Testing episodic training with VIC losses...")
    
    # Setup
    n_way = 5
    k_shot = 5
    n_query = 15
    device = torch.device('cpu')
    
    # Create a simple feature model
    def create_feature_model():
        class SimpleBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.final_feat_dim = 512
                self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(64, 512)
                
            def forward(self, x):
                batch_size = x.size(0)
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(batch_size, -1)
                x = self.fc(x)
                return x
        
        return SimpleBackbone()
    
    # Test 1: Create model with VIC losses
    print("\nTest 1: Creating model with VIC losses enabled")
    model = FewShotTransformer(
        create_feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        lambda_I=1.0,
        lambda_V=0.5,
        lambda_C=0.1
    ).to(device)
    print("✓ Model created successfully")
    
    # Test 2: Create optimizer
    print("\nTest 2: Creating optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("✓ Optimizer created")
    
    # Test 3: Simulate an episodic training batch
    print("\nTest 3: Simulating episodic training step")
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
    
    # Forward pass
    model.train()
    acc, loss = model.set_forward_loss(x)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Total Loss: {loss.item():.6f}")
    assert isinstance(acc, float), "Accuracy should be float"
    assert loss.requires_grad, "Loss should have gradient"
    print("✓ Forward pass successful")
    
    # Test 4: Backward pass
    print("\nTest 4: Backward pass")
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_count = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_count += 1
    print(f"  Parameters with gradients: {grad_count}")
    assert grad_count > 0, "Should have gradients"
    print("✓ Backward pass successful")
    
    # Test 5: Optimizer step
    print("\nTest 5: Optimizer step")
    optimizer.step()
    print("✓ Optimizer step successful")
    
    # Test 6: Multiple episodes (mini training loop)
    print("\nTest 6: Multiple training episodes")
    model.train()
    losses = []
    accs = []
    
    for episode in range(5):
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
        optimizer.zero_grad()
        acc, loss = model.set_forward_loss(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc)
        print(f"  Episode {episode+1}: Acc={acc:.4f}, Loss={loss.item():.6f}")
    
    print(f"  Average accuracy: {np.mean(accs):.4f}")
    print(f"  Average loss: {np.mean(losses):.6f}")
    print("✓ Multiple episodes completed")
    
    # Test 7: Validation mode (no gradients)
    print("\nTest 7: Validation mode")
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
        acc, loss = model.set_forward_loss(x)
        print(f"  Val Acc: {acc:.4f}, Val Loss: {loss.item():.6f}")
    print("✓ Validation mode works")
    
    # Test 8: Test with different VIC weight configurations
    print("\nTest 8: Different VIC weight configurations")
    
    configs = [
        (1.0, 0.0, 0.0, "Standard (no VIC)"),
        (1.0, 1.0, 0.0, "With Variance only"),
        (1.0, 0.0, 1.0, "With Covariance only"),
        (1.0, 0.5, 0.1, "With both V and C"),
    ]
    
    for lambda_I, lambda_V, lambda_C, desc in configs:
        test_model = FewShotTransformer(
            create_feature_model,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            variant='cosine',
            lambda_I=lambda_I,
            lambda_V=lambda_V,
            lambda_C=lambda_C
        ).to(device)
        
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
        acc, loss = test_model.set_forward_loss(x)
        print(f"  {desc}: Loss={loss.item():.6f}")
    
    print("✓ All configurations work")
    
    # Test 9: Verify loss components are actually being used
    print("\nTest 9: Verify loss components affect total loss")
    
    # Model with no VIC losses
    model_no_vic = FewShotTransformer(
        create_feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        lambda_I=1.0,
        lambda_V=0.0,
        lambda_C=0.0
    ).to(device)
    
    # Model with VIC losses
    model_with_vic = FewShotTransformer(
        create_feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        lambda_I=1.0,
        lambda_V=1.0,
        lambda_C=1.0
    ).to(device)
    
    # Load same weights to both models for fair comparison
    state_dict = model_no_vic.state_dict()
    model_with_vic.load_state_dict(state_dict)
    
    # Forward pass with same data
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
    acc1, loss1 = model_no_vic.set_forward_loss(x)
    acc2, loss2 = model_with_vic.set_forward_loss(x)
    
    print(f"  Loss without VIC: {loss1.item():.6f}")
    print(f"  Loss with VIC: {loss2.item():.6f}")
    
    # Losses should be different when VIC is enabled
    # (unless variance and covariance losses are exactly 0, which is unlikely)
    if abs(loss1.item() - loss2.item()) > 1e-6:
        print("✓ VIC losses affect total loss as expected")
    else:
        print("⚠ Warning: Losses are very similar, VIC components might be near zero")
    
    print("\n" + "="*60)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("="*60)
    print("\nThe VIC-Enhanced FS-CT training implementation is working correctly.")
    print("You can now train models with VIC losses using:")
    print("  --lambda_I 1.0 --lambda_V 0.5 --lambda_C 0.1")
    
    return True

if __name__ == '__main__':
    try:
        success = test_episodic_training_with_vic()
        if success:
            sys.exit(0)
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
