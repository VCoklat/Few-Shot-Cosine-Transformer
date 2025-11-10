"""
Unit tests for VIC Regularization and Enhanced Transformer components
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

from methods.vic_regularization import VICRegularization, MahalanobisClassifier, DynamicWeightController
from methods.enhanced_transformer import EnhancedFewShotTransformer, CosineAttention, CosineAttentionBlock

def test_vic_regularization():
    """Test VIC regularization computation"""
    print("Testing VIC Regularization...")
    
    # Create dummy embeddings
    n_way = 5
    k_shot = 5
    d = 64
    
    support_embeddings = torch.randn(n_way, k_shot, d)
    prototypes = torch.randn(n_way, d)
    
    vic = VICRegularization(feature_dim=d)
    
    # Test variance loss
    loss_V, loss_C = vic(support_embeddings, prototypes)
    
    assert loss_V.item() >= 0, "Variance loss should be non-negative"
    assert loss_C.item() >= 0, "Covariance loss should be non-negative"
    
    print(f"  ✓ Variance loss: {loss_V.item():.4f}")
    print(f"  ✓ Covariance loss: {loss_C.item():.4f}")
    print("  ✓ VIC Regularization test passed\n")

def test_mahalanobis_classifier():
    """Test Mahalanobis distance classifier"""
    print("Testing Mahalanobis Classifier...")
    
    n_way = 5
    k_shot = 5
    n_query = 10
    d = 64
    
    queries = torch.randn(n_query, d)
    support_embeddings = torch.randn(n_way, k_shot, d)
    prototypes = torch.randn(n_way, d)
    
    classifier = MahalanobisClassifier(shrinkage_param=0.1)
    
    # Compute distances
    scores = classifier(queries, support_embeddings, prototypes)
    
    assert scores.shape == (n_query, n_way), f"Expected shape {(n_query, n_way)}, got {scores.shape}"
    assert torch.isfinite(scores).all(), "All scores should be finite"
    
    print(f"  ✓ Output shape: {scores.shape}")
    print(f"  ✓ Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print("  ✓ Mahalanobis Classifier test passed\n")

def test_dynamic_weight_controller():
    """Test dynamic weight controller"""
    print("Testing Dynamic Weight Controller...")
    
    # Test uncertainty weighting
    controller = DynamicWeightController(strategy='uncertainty')
    
    loss_I = torch.tensor(2.5)
    loss_V = torch.tensor(0.3)
    loss_C = torch.tensor(0.4)
    
    total_loss, loss_dict = controller.compute_weighted_loss(loss_I, loss_V, loss_C)
    
    assert total_loss.item() > 0, "Total loss should be positive"
    assert 'loss_I' in loss_dict, "Loss dict should contain loss_I"
    assert 'weight_I' in loss_dict, "Loss dict should contain weight_I"
    
    print(f"  ✓ Total loss: {total_loss.item():.4f}")
    print(f"  ✓ Weights: I={loss_dict['weight_I']:.4f}, V={loss_dict['weight_V']:.4f}, C={loss_dict['weight_C']:.4f}")
    print("  ✓ Dynamic Weight Controller test passed\n")

def test_cosine_attention():
    """Test cosine attention mechanism"""
    print("Testing Cosine Attention...")
    
    batch = 2
    n_q = 10
    n_k = 5
    dim = 64
    heads = 4
    dim_head = 16
    
    q = torch.randn(batch, n_q, dim)
    k = torch.randn(batch, n_k, dim)
    v = torch.randn(batch, n_k, dim)
    
    # Test cosine variant
    attn_cosine = CosineAttention(dim, heads=heads, dim_head=dim_head, variant='cosine')
    out_cosine = attn_cosine(q, k, v)
    
    assert out_cosine.shape == (batch, n_q, dim), f"Expected shape {(batch, n_q, dim)}, got {out_cosine.shape}"
    
    # Test softmax variant
    attn_softmax = CosineAttention(dim, heads=heads, dim_head=dim_head, variant='softmax')
    out_softmax = attn_softmax(q, k, v)
    
    assert out_softmax.shape == (batch, n_q, dim), f"Expected shape {(batch, n_q, dim)}, got {out_softmax.shape}"
    
    print(f"  ✓ Cosine attention output shape: {out_cosine.shape}")
    print(f"  ✓ Softmax attention output shape: {out_softmax.shape}")
    print("  ✓ Cosine Attention test passed\n")

def test_attention_block():
    """Test cosine attention block with FFN"""
    print("Testing Cosine Attention Block...")
    
    batch = 2
    n_q = 10
    n_k = 5
    dim = 64
    
    q = torch.randn(batch, n_q, dim)
    k = torch.randn(batch, n_k, dim)
    v = torch.randn(batch, n_k, dim)
    
    block = CosineAttentionBlock(dim, heads=4, dim_head=16, mlp_dim=128, variant='cosine')
    out = block(q, k, v)
    
    assert out.shape == q.shape, f"Expected shape {q.shape}, got {out.shape}"
    
    print(f"  ✓ Output shape: {out.shape}")
    print("  ✓ Attention Block test passed\n")

def test_enhanced_transformer_forward():
    """Test enhanced transformer forward pass"""
    print("Testing Enhanced Transformer Forward Pass...")
    
    n_way = 5
    k_shot = 5
    n_query = 10
    
    # Create a dummy feature extractor
    class DummyFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.final_feat_dim = 64
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            return x.view(x.size(0), -1)
    
    def feature_model():
        return DummyFeatureExtractor()
    
    # Create model
    model = EnhancedFewShotTransformer(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        depth=2,
        heads=4,
        dim_head=16,
        use_vic=True
    )
    
    # Create dummy input (n_way, k_shot + n_query, C, H, W)
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Forward pass
    scores = model.set_forward(x)
    
    expected_shape = (n_way * n_query, n_way)
    assert scores.shape == expected_shape, f"Expected shape {expected_shape}, got {scores.shape}"
    
    print(f"  ✓ Output shape: {scores.shape}")
    print(f"  ✓ Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print("  ✓ Enhanced Transformer Forward test passed\n")

def test_enhanced_transformer_loss():
    """Test enhanced transformer with loss computation"""
    print("Testing Enhanced Transformer with Loss...")
    
    n_way = 5
    k_shot = 5
    n_query = 10
    
    class DummyFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.final_feat_dim = 64
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            return x.view(x.size(0), -1)
    
    def feature_model():
        return DummyFeatureExtractor()
    
    model = EnhancedFewShotTransformer(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        use_vic=True
    )
    
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Forward pass with loss
    acc, loss = model.set_forward_loss(x)
    
    assert 0 <= acc <= 1, f"Accuracy should be in [0, 1], got {acc}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    assert torch.isfinite(loss), "Loss should be finite"
    
    print(f"  ✓ Accuracy: {acc:.4f}")
    print(f"  ✓ Loss: {loss.item():.4f}")
    print("  ✓ Enhanced Transformer Loss test passed\n")

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running VIC Regularization and Enhanced Transformer Tests")
    print("="*60 + "\n")
    
    try:
        test_vic_regularization()
        test_mahalanobis_classifier()
        test_dynamic_weight_controller()
        test_cosine_attention()
        test_attention_block()
        test_enhanced_transformer_forward()
        test_enhanced_transformer_loss()
        
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
