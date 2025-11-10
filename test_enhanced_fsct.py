"""
Test script for EnhancedFSCT implementation
"""

import torch
import numpy as np
from methods.enhanced_fsct import EnhancedFSCT, CosineEncoderBlock
import backbone

def test_cosine_encoder_block():
    """Test CosineEncoderBlock with proper shapes"""
    print("Testing CosineEncoderBlock...")
    
    dim = 512
    n_query = 25  # 5 way * 5 query
    n_way = 5
    
    block = CosineEncoderBlock(dim=dim, heads=4, dim_head=64, mlp_dim=512)
    
    # Create test data
    queries = torch.randn(n_query, dim)
    prototypes = torch.randn(n_way, dim)
    
    # Forward pass
    output = block(queries, prototypes)
    
    assert output.shape == (n_query, dim), f"Expected shape {(n_query, dim)}, got {output.shape}"
    print(f"✓ CosineEncoderBlock output shape: {output.shape}")
    print()

def test_enhanced_fsct_forward():
    """Test EnhancedFSCT forward pass"""
    print("Testing EnhancedFSCT forward pass...")
    
    # Parameters
    n_way = 5
    k_shot = 5
    n_query = 8
    
    # Create a simple feature extractor
    def feature_model():
        # Use Conv4 for testing (simpler than ResNet)
        return backbone.Conv4('miniImagenet', flatten=True)
    
    # Initialize model
    model = EnhancedFSCT(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        depth=2,
        heads=4,
        dim_head=64,
        mlp_dim=512,
        use_uncertainty_weighting=True
    )
    
    model.eval()
    
    # Create dummy input: (n_way, k_shot + n_query, C, H, W)
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Forward pass
    with torch.no_grad():
        scores = model.set_forward(x)
    
    expected_shape = (n_way * n_query, n_way)
    assert scores.shape == expected_shape, f"Expected shape {expected_shape}, got {scores.shape}"
    print(f"✓ EnhancedFSCT scores shape: {scores.shape}")
    
    # Check that scores are valid
    assert not torch.isnan(scores).any(), "NaN values in scores"
    assert not torch.isinf(scores).any(), "Inf values in scores"
    print("✓ Scores are finite")
    print()

def test_enhanced_fsct_loss():
    """Test EnhancedFSCT with loss computation"""
    print("Testing EnhancedFSCT with loss...")
    
    # Parameters
    n_way = 5
    k_shot = 1  # Test with 1-shot
    n_query = 8
    
    def feature_model():
        return backbone.Conv4('miniImagenet', flatten=True)
    
    # Test with uncertainty weighting
    model = EnhancedFSCT(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        depth=2,
        heads=4,
        dim_head=64,
        mlp_dim=512,
        use_uncertainty_weighting=True
    )
    
    model.train()
    
    # Create dummy input
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Forward pass with loss
    acc, loss = model.set_forward_loss(x)
    
    print(f"✓ Accuracy: {acc:.4f}")
    print(f"✓ Loss: {loss.item():.4f}")
    
    # Check that loss is valid
    assert not torch.isnan(loss).any(), "NaN values in loss"
    assert not torch.isinf(loss).any(), "Inf values in loss"
    assert loss.item() > 0, "Loss should be positive"
    assert 0 <= acc <= 1, f"Accuracy should be in [0, 1], got {acc}"
    print("✓ Loss and accuracy are valid")
    
    # Test backward pass
    loss.backward()
    print("✓ Backward pass successful")
    print()

def test_vic_losses():
    """Test VIC loss computation"""
    print("Testing VIC losses...")
    
    n_way = 5
    k_shot = 5
    n_query = 8
    
    def feature_model():
        return backbone.Conv4('miniImagenet', flatten=True)
    
    model = EnhancedFSCT(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        use_uncertainty_weighting=False  # Test with fixed weights
    )
    
    # Create test embeddings
    d = 1600  # Conv4 output dim
    embeddings = torch.randn(n_way * (k_shot + 1), d)
    
    # Test variance loss
    loss_V = model.compute_variance_loss(embeddings)
    print(f"✓ Variance loss: {loss_V.item():.4f}")
    assert not torch.isnan(loss_V), "Variance loss is NaN"
    
    # Test covariance loss
    loss_C = model.compute_covariance_loss(embeddings)
    print(f"✓ Covariance loss: {loss_C.item():.4f}")
    assert not torch.isnan(loss_C), "Covariance loss is NaN"
    print()

def test_mahalanobis_distance():
    """Test Mahalanobis distance computation"""
    print("Testing Mahalanobis distance...")
    
    n_way = 5
    k_shot = 5
    n_query = 8
    d = 1600
    
    def feature_model():
        return backbone.Conv4('miniImagenet', flatten=True)
    
    model = EnhancedFSCT(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query
    )
    
    # Create test data
    query_features = torch.randn(n_way * n_query, d)
    prototypes = torch.randn(n_way, d)
    support_features = torch.randn(n_way, k_shot, d)
    
    # Compute distances
    distances = model.mahalanobis_distance(query_features, prototypes, support_features)
    
    expected_shape = (n_way * n_query, n_way)
    assert distances.shape == expected_shape, f"Expected shape {expected_shape}, got {distances.shape}"
    print(f"✓ Mahalanobis distances shape: {distances.shape}")
    
    # Check that distances are non-negative
    assert (distances >= 0).all(), "Distances should be non-negative"
    assert not torch.isnan(distances).any(), "NaN in distances"
    print("✓ Distances are valid")
    print()

def test_learnable_prototypes():
    """Test learnable weighted prototypes"""
    print("Testing learnable weighted prototypes...")
    
    n_way = 5
    k_shot = 5
    n_query = 8
    d = 1600
    
    def feature_model():
        return backbone.Conv4('miniImagenet', flatten=True)
    
    model = EnhancedFSCT(
        feature_model,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query
    )
    
    # Create test support features
    z_support = torch.randn(n_way, k_shot, d)
    
    # Compute weighted prototypes
    z_proto = model.compute_weighted_prototypes(z_support)
    
    expected_shape = (n_way, d)
    assert z_proto.shape == expected_shape, f"Expected shape {expected_shape}, got {z_proto.shape}"
    print(f"✓ Prototype shape: {z_proto.shape}")
    
    # Check that weights sum to 1 across shots
    weights = torch.softmax(model.proto_weight, dim=1)
    weight_sums = weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "Weights don't sum to 1"
    print(f"✓ Weights sum to 1 per class")
    print(f"  Sample weights for class 0: {weights[0].squeeze().tolist()}")
    print()

if __name__ == '__main__':
    print("="*60)
    print("Enhanced FSCT Component Tests")
    print("="*60)
    print()
    
    try:
        test_cosine_encoder_block()
        test_learnable_prototypes()
        test_vic_losses()
        test_mahalanobis_distance()
        test_enhanced_fsct_forward()
        test_enhanced_fsct_loss()
        
        print("="*60)
        print("✓ All tests passed!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
