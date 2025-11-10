"""
Integration test to verify DV-FSCT can be instantiated and run
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.dvfsct import DVFSCT
from backbone import Conv4, ResNet18

device = torch.device('cpu')

def test_instantiation():
    """Test that DVFSCT can be instantiated with different backbones"""
    print("\n=== Testing Model Instantiation ===")
    
    # Test with Conv4 backbone
    print("\n1. Testing with Conv4 backbone...")
    model_conv4 = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16,
        depth=1, heads=8, dim_head=64, mlp_dim=512,
        lambda_vic=0.1
    ).to(device)
    print(f"✓ Conv4 model instantiated successfully")
    print(f"  Feature dim: {model_conv4.feat_dim}")
    print(f"  Proto weights shape: {model_conv4.proto_weight.shape}")
    
    # Test with ResNet18 backbone
    print("\n2. Testing with ResNet18 backbone...")
    try:
        model_resnet = DVFSCT(
            model_func=lambda: ResNet18(feti=False, dataset='miniImagenet', flatten=True),
            n_way=5, k_shot=5, n_query=16,
            depth=1, heads=8, dim_head=64, mlp_dim=512,
            lambda_vic=0.1
        ).to(device)
        print(f"✓ ResNet18 model instantiated successfully")
        print(f"  Feature dim: {model_resnet.feat_dim}")
    except Exception as e:
        print(f"⚠ ResNet18 instantiation failed (expected if no pretrained weights): {e}")
    
    return model_conv4


def test_episodic_forward():
    """Test that model can process an episodic batch"""
    print("\n=== Testing Episodic Forward Pass ===")
    
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16,
        depth=1, heads=8, dim_head=64, mlp_dim=512,
        lambda_vic=0.1
    ).to(device)
    
    # Create a full episode (5-way, 5-shot, 16 queries)
    # Shape: [N_way, K_shot + N_query, C, H, W]
    episode = torch.randn(5, 21, 3, 84, 84)
    
    print(f"Episode shape: {episode.shape}")
    print(f"  N-way: 5, K-shot: 5, N-query: 16")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        scores = model.set_forward(episode, is_feature=False)
    
    print(f"✓ Forward pass successful")
    print(f"  Output scores shape: {scores.shape}")
    print(f"  Expected: (80, 5) [N_query * N_way, N_way]")
    assert scores.shape == (80, 5), f"Unexpected output shape: {scores.shape}"
    
    # Test loss computation
    model.train()
    acc, loss = model.set_forward_loss(episode)
    print(f"✓ Loss computation successful")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Loss: {loss.item():.4f}")


def test_training_step():
    """Test that model can perform a training step"""
    print("\n=== Testing Training Step ===")
    
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16,
        depth=1, heads=8, dim_head=64, mlp_dim=512,
        lambda_vic=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Create episode
    episode = torch.randn(5, 21, 3, 84, 84)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    acc, loss = model.set_forward_loss(episode)
    loss.backward()
    
    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    optimizer.step()
    
    print(f"✓ Training step successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Gradients computed for {len(grad_norms)} parameters")
    print(f"  Sample gradient norms:")
    for name, norm in list(grad_norms.items())[:3]:
        print(f"    {name}: {norm:.6f}")


def test_different_shots():
    """Test that model works with different shot numbers"""
    print("\n=== Testing Different Shot Numbers ===")
    
    for k_shot in [1, 5, 10]:
        print(f"\nTesting {k_shot}-shot...")
        model = DVFSCT(
            model_func=lambda: Conv4('miniImagenet', flatten=True),
            n_way=5, k_shot=k_shot, n_query=16,
            depth=1, heads=8, dim_head=64, mlp_dim=512,
            lambda_vic=0.1
        ).to(device)
        
        # Create episode
        episode = torch.randn(5, k_shot + 16, 3, 84, 84)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            scores = model.set_forward(episode, is_feature=False)
        
        print(f"  ✓ {k_shot}-shot: output shape {scores.shape}")


def test_vic_components():
    """Test individual VIC loss components"""
    print("\n=== Testing VIC Loss Components ===")
    
    model = DVFSCT(
        model_func=lambda: Conv4('miniImagenet', flatten=True),
        n_way=5, k_shot=5, n_query=16,
        depth=1, heads=8, dim_head=64, mlp_dim=512,
        lambda_vic=0.1
    ).to(device)
    
    # Create episode
    episode = torch.randn(5, 21, 3, 84, 84)
    
    # Get features
    z_support, z_query = model.parse_feature(episode, is_feature=False)
    z_support = z_support.contiguous().view(5, 5, -1)
    
    # Test hardness computation
    h_bar, h_classes = model.compute_hardness_scores(z_support)
    print(f"Hardness score: {h_bar.item():.4f}")
    print(f"  Class hardness: {h_classes.detach().cpu().numpy()}")
    
    # Test VIC losses
    V = model.vic_variance_loss(z_support)
    C = model.vic_covariance_loss(z_support)
    print(f"✓ VIC components computed")
    print(f"  Variance loss: {V.item():.4f}")
    print(f"  Covariance loss: {C.item():.4f}")
    
    # Test dynamic weights
    alpha_V = 0.5 + 0.5 * h_bar.item()
    alpha_C = 0.5 + 0.5 * h_bar.item()
    print(f"  Dynamic weights: alpha_V={alpha_V:.4f}, alpha_C={alpha_C:.4f}")


def run_integration_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("DV-FSCT Integration Tests")
    print("="*60)
    
    try:
        test_instantiation()
        test_episodic_forward()
        test_training_step()
        test_different_shots()
        test_vic_components()
        
        print("\n" + "="*60)
        print("✓ All integration tests passed!")
        print("="*60)
        return True
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ Integration test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
