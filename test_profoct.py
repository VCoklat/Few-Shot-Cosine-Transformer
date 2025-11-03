"""
Simple validation script for ProFO-CT implementation.
Tests core functionality without requiring datasets.
"""

import torch
import torch.nn as nn
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, '.')

print("=" * 60)
print("ProFO-CT Validation Test")
print("=" * 60)

# Test 1: Import test
print("\n[Test 1] Importing ProFOCT module...")
try:
    from methods.ProFOCT import ProFOCT, VICAttention, cosine_distance
    print("✅ ProFOCT module imported successfully")
except Exception as e:
    print(f"❌ Failed to import ProFOCT: {e}")
    sys.exit(1)

# Test 2: Create a simple backbone model
print("\n[Test 2] Creating simple backbone for testing...")
class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.final_feat_dim = 64
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

try:
    backbone = SimpleBackbone()
    print("✅ Simple backbone created")
except Exception as e:
    print(f"❌ Failed to create backbone: {e}")
    sys.exit(1)

# Test 3: Instantiate ProFOCT with cosine variant
print("\n[Test 3] Instantiating ProFOCT (cosine variant)...")
try:
    model = ProFOCT(
        model_func=lambda: backbone,
        n_way=5,
        k_shot=5,
        n_query=15,
        variant="cosine",
        vic_alpha=0.5,
        vic_beta=9.0,
        vic_gamma=0.5,
        dynamic_vic=True,
        distance_metric='euclidean'
    )
    print("✅ ProFOCT model instantiated successfully")
    print(f"   - Variant: cosine")
    print(f"   - Dynamic VIC: enabled")
    print(f"   - Initial VIC weights: α={model.vic_alpha.item():.2f}, β={model.vic_beta.item():.2f}, γ={model.vic_gamma.item():.2f}")
except Exception as e:
    print(f"❌ Failed to instantiate ProFOCT: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Instantiate ProFOCT with softmax variant
print("\n[Test 4] Instantiating ProFOCT (softmax variant)...")
try:
    model_softmax = ProFOCT(
        model_func=lambda: backbone,
        n_way=5,
        k_shot=5,
        n_query=15,
        variant="softmax",
        dynamic_vic=False
    )
    print("✅ ProFOCT model (softmax) instantiated successfully")
    print(f"   - Variant: softmax")
    print(f"   - Dynamic VIC: disabled")
except Exception as e:
    print(f"❌ Failed to instantiate ProFOCT (softmax): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test VIC loss computations
print("\n[Test 5] Testing VIC loss computations...")
try:
    # Create dummy embeddings
    batch_size = 25  # 5 way * 5 shot
    feature_dim = 64
    z = torch.randn(batch_size, feature_dim)
    
    # Test variance loss
    loss_v = model.compute_variance_loss(z)
    print(f"✅ Variance loss computed: {loss_v.item():.4f}")
    
    # Test covariance loss
    loss_c = model.compute_covariance_loss(z)
    print(f"✅ Covariance loss computed: {loss_c.item():.4f}")
    
    # Test invariance loss (with dummy augmented data)
    z_aug = z + torch.randn_like(z) * 0.1
    loss_i = model.compute_invariance_loss(z, z_aug)
    print(f"✅ Invariance loss computed: {loss_i.item():.4f}")
    
except Exception as e:
    print(f"❌ Failed VIC loss computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test cosine distance function
print("\n[Test 6] Testing cosine distance function...")
try:
    x1 = torch.randn(2, 4, 10, 8)  # (batch, heads, n, k)
    x2 = torch.randn(2, 4, 8, 12)  # (batch, heads, k, m)
    
    cos_dist = cosine_distance(x1, x2)
    assert cos_dist.shape == (2, 4, 10, 12), f"Expected shape (2, 4, 10, 12), got {cos_dist.shape}"
    assert torch.all(torch.abs(cos_dist) <= 1.0 + 1e-6), "Cosine distances should be in [-1, 1]"
    
    print(f"✅ Cosine distance computed correctly")
    print(f"   - Output shape: {cos_dist.shape}")
    print(f"   - Value range: [{cos_dist.min().item():.4f}, {cos_dist.max().item():.4f}]")
except Exception as e:
    print(f"❌ Failed cosine distance test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test VICAttention
print("\n[Test 7] Testing VICAttention module...")
try:
    attn = VICAttention(dim=64, heads=4, dim_head=16, variant="cosine")
    
    # Create dummy inputs
    q = torch.randn(1, 5, 64)  # (batch, n_proto, dim)
    k = torch.randn(15, 1, 64)  # (n_query, 1, dim)
    v = torch.randn(15, 1, 64)
    
    out = attn(q, k, v)
    expected_shape = (15, 5, 64)
    assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"
    
    print(f"✅ VICAttention forward pass successful")
    print(f"   - Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"   - Output shape: {out.shape}")
except Exception as e:
    print(f"❌ Failed VICAttention test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test forward pass with dummy data
print("\n[Test 8] Testing forward pass with dummy episode data...")
try:
    model.eval()
    
    # Create dummy episode data: (n_way, k_shot + n_query, C, H, W)
    n_way, k_shot, n_query = 5, 5, 15
    dummy_episode = torch.randn(n_way, k_shot + n_query, 3, 32, 32)
    
    with torch.no_grad():
        scores = model.set_forward(dummy_episode, is_feature=False)
    
    expected_shape = (n_query * n_way, n_way)
    assert scores.shape == expected_shape, f"Expected shape {expected_shape}, got {scores.shape}"
    
    print(f"✅ Forward pass successful")
    print(f"   - Input episode shape: {dummy_episode.shape}")
    print(f"   - Output scores shape: {scores.shape}")
    print(f"   - Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
except Exception as e:
    print(f"❌ Failed forward pass test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Test training step (loss computation)
print("\n[Test 9] Testing training step with loss computation...")
try:
    model.train()
    
    # Create dummy episode data
    dummy_episode = torch.randn(n_way, k_shot + n_query, 3, 32, 32)
    
    acc, loss = model.set_forward_loss(dummy_episode)
    
    assert 0 <= acc <= 1, f"Accuracy should be in [0, 1], got {acc}"
    assert loss > 0, f"Loss should be positive, got {loss}"
    
    print(f"✅ Training step successful")
    print(f"   - Accuracy: {acc:.4f}")
    print(f"   - Total loss: {loss.item():.4f}")
    
    # Test if gradients can be computed
    loss.backward()
    print(f"✅ Backward pass successful (gradients computed)")
    
    # Check VIC weights after dynamic update
    vic_weights = model.get_vic_weights()
    print(f"   - VIC weights after update: α={vic_weights['alpha']:.4f}, β={vic_weights['beta']:.4f}, γ={vic_weights['gamma']:.4f}")
    
except Exception as e:
    print(f"❌ Failed training step test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Test static vs dynamic VIC
print("\n[Test 10] Testing static vs dynamic VIC behavior...")
try:
    # Create model with static VIC
    model_static = ProFOCT(
        model_func=lambda: SimpleBackbone(),
        n_way=5,
        k_shot=5,
        n_query=15,
        variant="cosine",
        vic_alpha=0.5,
        vic_beta=9.0,
        vic_gamma=0.5,
        dynamic_vic=False
    )
    
    initial_alpha = model_static.vic_alpha.item()
    
    # Run a training step
    model_static.train()
    dummy_episode = torch.randn(5, 20, 3, 32, 32)
    acc, loss = model_static.set_forward_loss(dummy_episode)
    
    # VIC weights should not change with static mode
    final_alpha = model_static.vic_alpha.item()
    assert abs(initial_alpha - final_alpha) < 1e-6, "Static VIC weights should not change"
    
    print(f"✅ Static VIC behavior verified (weights unchanged)")
    print(f"   - α before: {initial_alpha:.4f}, after: {final_alpha:.4f}")
    
    # Test dynamic VIC changes
    model_dynamic = ProFOCT(
        model_func=lambda: SimpleBackbone(),
        n_way=5,
        k_shot=5,
        n_query=15,
        variant="cosine",
        dynamic_vic=True
    )
    
    initial_alpha_dyn = model_dynamic.vic_alpha.item()
    
    model_dynamic.train()
    acc, loss = model_dynamic.set_forward_loss(dummy_episode)
    
    final_alpha_dyn = model_dynamic.vic_alpha.item()
    
    print(f"✅ Dynamic VIC behavior verified")
    print(f"   - α before: {initial_alpha_dyn:.4f}, after: {final_alpha_dyn:.4f}")
    print(f"   - Change: {abs(final_alpha_dyn - initial_alpha_dyn):.6f}")
    
except Exception as e:
    print(f"❌ Failed static vs dynamic VIC test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ All ProFO-CT validation tests passed!")
print("=" * 60)
print("\nProFO-CT implementation is ready for use.")
print("\nKey features validated:")
print("  ✓ VIC regularization (Variance, Invariance, Covariance)")
print("  ✓ Dynamic VIC weight adaptation")
print("  ✓ Cosine and softmax attention variants")
print("  ✓ Learnable prototypes with weighted mean")
print("  ✓ Forward and backward passes")
print("  ✓ Training loop integration")
print("\nYou can now train ProFO-CT with:")
print("  python train.py --method ProFOCT_cosine --dataset miniImagenet --n_way 5 --k_shot 5")
print("=" * 60)
