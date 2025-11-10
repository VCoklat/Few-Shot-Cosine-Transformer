"""
Test script for Dynamic-VIC Few-Shot Cosine Transformer (DV-FSCT)

This script validates the implementation with dummy data to ensure:
1. VIC loss computation works correctly
2. Dynamic weight generation based on hardness works
3. Forward pass completes without errors
4. Model can be trained for a few steps
"""

import torch
import torch.nn as nn
import numpy as np
from methods.dv_fsct import DVFSCT, CosineAttention
import backbone

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def create_dummy_backbone():
    """Create a simple dummy backbone for testing"""
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 512)
            self.final_feat_dim = 512
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    return DummyBackbone


def test_vic_loss():
    """Test VIC loss computation"""
    print("\n" + "="*60)
    print("Test 1: VIC Loss Computation")
    print("="*60)
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    model = DVFSCT(
        create_dummy_backbone(),
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        use_mixed_precision=False  # Disable for CPU testing
    ).to(device)
    
    # Create dummy data
    z_support = torch.randn(n_way * k_shot, 512).to(device)
    z_query = torch.randn(n_way * n_query, 512).to(device)
    y_support = torch.from_numpy(np.repeat(range(n_way), k_shot)).to(device)
    y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).to(device)
    
    # Test dynamic weight computation
    z_support_reshaped = z_support.view(n_way, k_shot, -1)
    alpha_V, alpha_I, alpha_C, h_bar = model.compute_dynamic_weights(z_support_reshaped)
    
    print(f"Dynamic weights computed:")
    print(f"  alpha_V: {alpha_V:.4f}")
    print(f"  alpha_I: {alpha_I:.4f}")
    print(f"  alpha_C: {alpha_C:.4f}")
    print(f"  hardness (h_bar): {h_bar:.4f}")
    
    # Test VIC loss computation
    vic_loss, V, I, C = model.vic_loss(z_support, y_support, z_query, y_query,
                                       alpha_V, alpha_I, alpha_C)
    
    print(f"\nVIC loss components:")
    print(f"  Variance (V): {V.item():.4f}")
    print(f"  Invariance (I): {I.item():.4f}")
    print(f"  Covariance (C): {C.item():.4f}")
    print(f"  Total VIC loss: {vic_loss.item():.4f}")
    
    assert not torch.isnan(vic_loss), "VIC loss contains NaN"
    assert not torch.isinf(vic_loss), "VIC loss contains Inf"
    print("✓ VIC loss computation passed")
    
    return True


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\n" + "="*60)
    print("Test 2: Forward Pass")
    print("="*60)
    
    n_way = 5
    k_shot = 5
    n_query = 15
    batch_size = n_way * (k_shot + n_query)
    
    model = DVFSCT(
        create_dummy_backbone(),
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        use_mixed_precision=False
    ).to(device)
    
    # Create dummy images (N-way, K+Q shots, C, H, W)
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        scores = model.set_forward(x, is_feature=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Expected shape: [{n_way * n_query}, {n_way}]")
    
    assert scores.shape == (n_way * n_query, n_way), \
        f"Output shape mismatch: {scores.shape} vs {(n_way * n_query, n_way)}"
    assert not torch.isnan(scores).any(), "Scores contain NaN"
    assert not torch.isinf(scores).any(), "Scores contain Inf"
    
    print("✓ Forward pass test passed")
    
    return True


def test_training_step():
    """Test a single training step"""
    print("\n" + "="*60)
    print("Test 3: Training Step")
    print("="*60)
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    model = DVFSCT(
        create_dummy_backbone(),
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        vic_lambda=0.1,
        use_mixed_precision=False
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Create dummy data
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    acc, loss = model.set_forward_loss(x)
    
    print(f"Training metrics:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            print(f"  Gradient norm for {name}: {grad_norm:.6f}")
            break
    
    assert has_grad, "No gradients computed"
    assert not torch.isnan(loss), "Loss contains NaN"
    assert not torch.isinf(loss), "Loss contains Inf"
    
    optimizer.step()
    
    print("✓ Training step test passed")
    
    return True


def test_cosine_attention():
    """Test cosine attention mechanism"""
    print("\n" + "="*60)
    print("Test 4: Cosine Attention")
    print("="*60)
    
    dim = 512
    heads = 8
    dim_head = 64
    
    attn = CosineAttention(dim, heads=heads, dim_head=dim_head).to(device)
    
    # Create dummy tensors
    q = torch.randn(1, 5, dim).to(device)  # [batch, n_proto, dim]
    k = torch.randn(1, 75, dim).to(device)  # [batch, n_query, dim]
    v = torch.randn(1, 75, dim).to(device)  # [batch, n_query, dim]
    
    # Forward pass
    out = attn(q, k, v)
    
    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected output shape: {q.shape}")
    
    assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    
    print("✓ Cosine attention test passed")
    
    return True


def test_multiple_episodes():
    """Test multiple training episodes"""
    print("\n" + "="*60)
    print("Test 5: Multiple Training Episodes")
    print("="*60)
    
    n_way = 5
    k_shot = 5
    n_query = 15
    n_episodes = 3
    
    model = DVFSCT(
        create_dummy_backbone(),
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        vic_lambda=0.1,
        use_mixed_precision=False
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    losses = []
    accs = []
    
    model.train()
    for episode in range(n_episodes):
        # Create dummy episode data
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84).to(device)
        
        optimizer.zero_grad()
        acc, loss = model.set_forward_loss(x)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        accs.append(acc)
        
        print(f"Episode {episode+1}/{n_episodes}: Loss={loss.item():.4f}, Acc={acc*100:.2f}%")
    
    print(f"\nAverage Loss: {np.mean(losses):.4f}")
    print(f"Average Accuracy: {np.mean(accs)*100:.2f}%")
    
    assert all(not np.isnan(l) for l in losses), "Loss contains NaN"
    assert all(not np.isinf(l) for l in losses), "Loss contains Inf"
    
    print("✓ Multiple episodes test passed")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("DV-FSCT Implementation Tests")
    print("="*60)
    
    tests = [
        ("VIC Loss Computation", test_vic_loss),
        ("Forward Pass", test_forward_pass),
        ("Training Step", test_training_step),
        ("Cosine Attention", test_cosine_attention),
        ("Multiple Episodes", test_multiple_episodes),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAILED"))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(status == "PASSED" for _, status in results)
    print("\n" + "="*60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
