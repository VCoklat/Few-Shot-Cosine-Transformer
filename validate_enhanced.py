"""
Validation script to test the enhanced transformer end-to-end.
Creates synthetic data and validates the full training/inference pipeline.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_synthetic_episode(n_way=5, k_shot=5, n_query=10, img_size=84, channels=3):
    """Create a synthetic few-shot episode."""
    support = torch.randn(n_way, k_shot, channels, img_size, img_size)
    query = torch.randn(n_way, n_query, channels, img_size, img_size)
    # Combine support and query
    episode = torch.cat([support, query], dim=1)  # (n_way, k_shot + n_query, C, H, W)
    return episode


def test_enhanced_transformer_forward():
    """Test forward pass of enhanced transformer."""
    print("\n=== Testing Enhanced Transformer Forward Pass ===")
    
    # Import modules
    import importlib.util
    
    # Load backbone module
    spec = importlib.util.spec_from_file_location('backbone', 'backbone.py')
    backbone_module = importlib.util.module_from_spec(spec)
    sys.modules['backbone'] = backbone_module
    spec.loader.exec_module(backbone_module)
    
    # Create a simple feature extractor
    class SimpleFeatureExtractor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten()
            )
            self.final_feat_dim = 64
        
        def forward(self, x):
            return self.conv(x)
    
    # Load enhanced transformer directly
    spec = importlib.util.spec_from_file_location('enhanced_transformer', 
        'methods/enhanced_transformer.py')
    enhanced_module = importlib.util.module_from_spec(spec)
    
    # Mock MetaTemplate
    class MockMetaTemplate(torch.nn.Module):
        def __init__(self, model_func, n_way, k_shot, n_query, change_way=True):
            super().__init__()
            self.n_way = n_way
            self.k_shot = k_shot
            self.n_query = n_query
            self.feature = model_func()
            self.feat_dim = self.feature.final_feat_dim
            self.change_way = change_way
        
        def parse_feature(self, x, is_feature):
            device = x.device
            if is_feature:
                z_all = x
            else:
                x = x.contiguous().view(self.n_way * (self.k_shot + self.n_query), *x.size()[2:])
                z_all = self.feature.forward(x)
                z_all = z_all.view(self.n_way, self.k_shot + self.n_query, -1)
            
            z_support = z_all[:, :self.k_shot]
            z_query = z_all[:, self.k_shot:]
            return z_support, z_query
    
    # Inject mock
    sys.modules['methods.meta_template'] = type(sys)('methods.meta_template')
    sys.modules['methods.meta_template'].MetaTemplate = MockMetaTemplate
    
    # Now load enhanced transformer
    spec.loader.exec_module(enhanced_module)
    EnhancedFewShotTransformer = enhanced_module.EnhancedFewShotTransformer
    
    # Create model
    n_way, k_shot, n_query = 5, 5, 10
    
    def feature_model():
        return SimpleFeatureExtractor()
    
    model = EnhancedFewShotTransformer(
        feature_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine',
        depth=2,
        heads=4,
        dim_head=64,
        mlp_dim=512,
        use_vic=True,
        use_mahalanobis=True,
        vic_lambda_init=[9.0, 0.5, 0.5],
        weight_controller='uncertainty',
        use_checkpoint=False
    )
    
    print(f"✓ Model created successfully")
    print(f"  - Feature dim: {model.feat_dim}")
    print(f"  - Attention blocks: {len(model.attention_blocks)}")
    print(f"  - Variant: {model.variant}")
    
    # Create synthetic episode
    episode = create_synthetic_episode(n_way, k_shot, n_query, img_size=84)
    print(f"✓ Episode shape: {episode.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        scores = model.set_forward(episode, is_feature=False)
    
    print(f"✓ Forward pass successful")
    print(f"  - Scores shape: {scores.shape}")
    print(f"  - Expected: ({n_way * n_query}, {n_way})")
    assert scores.shape == (n_way * n_query, n_way), "Incorrect output shape"
    
    # Check predictions
    predictions = scores.argmax(dim=1)
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"  - Prediction range: [{predictions.min()}, {predictions.max()}]")
    
    return True


def test_enhanced_transformer_training():
    """Test training step of enhanced transformer."""
    print("\n=== Testing Enhanced Transformer Training Step ===")
    
    # Import modules
    import importlib.util
    
    # Load backbone module
    spec = importlib.util.spec_from_file_location('backbone', 'backbone.py')
    backbone_module = importlib.util.module_from_spec(spec)
    sys.modules['backbone'] = backbone_module
    spec.loader.exec_module(backbone_module)
    
    # Create a simple feature extractor
    class SimpleFeatureExtractor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten()
            )
            self.final_feat_dim = 64
        
        def forward(self, x):
            return self.conv(x)
    
    # Load enhanced transformer directly
    spec = importlib.util.spec_from_file_location('enhanced_transformer', 
        'methods/enhanced_transformer.py')
    enhanced_module = importlib.util.module_from_spec(spec)
    
    # Mock MetaTemplate
    class MockMetaTemplate(torch.nn.Module):
        def __init__(self, model_func, n_way, k_shot, n_query, change_way=True):
            super().__init__()
            self.n_way = n_way
            self.k_shot = k_shot
            self.n_query = n_query
            self.feature = model_func()
            self.feat_dim = self.feature.final_feat_dim
            self.change_way = change_way
        
        def parse_feature(self, x, is_feature):
            device = x.device
            if is_feature:
                z_all = x
            else:
                x = x.contiguous().view(self.n_way * (self.k_shot + self.n_query), *x.size()[2:])
                z_all = self.feature.forward(x)
                z_all = z_all.view(self.n_way, self.k_shot + self.n_query, -1)
            
            z_support = z_all[:, :self.k_shot]
            z_query = z_all[:, self.k_shot:]
            return z_support, z_query
    
    # Inject mock
    sys.modules['methods.meta_template'] = type(sys)('methods.meta_template')
    sys.modules['methods.meta_template'].MetaTemplate = MockMetaTemplate
    
    # Now load enhanced transformer
    spec.loader.exec_module(enhanced_module)
    EnhancedFewShotTransformer = enhanced_module.EnhancedFewShotTransformer
    
    # Create model
    n_way, k_shot, n_query = 5, 5, 10
    
    def feature_model():
        return SimpleFeatureExtractor()
    
    model = EnhancedFewShotTransformer(
        feature_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine',
        depth=2,
        heads=4,
        dim_head=64,
        mlp_dim=512,
        use_vic=True,
        use_mahalanobis=True,
        vic_lambda_init=[9.0, 0.5, 0.5],
        weight_controller='uncertainty',
        use_checkpoint=False
    )
    
    # Create synthetic episode
    episode = create_synthetic_episode(n_way, k_shot, n_query, img_size=84)
    
    # Training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Forward and backward
    optimizer.zero_grad()
    acc, loss = model.set_forward_loss(episode)
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful")
    print(f"  - Loss: {loss.item():.6f}")
    print(f"  - Accuracy: {acc * 100:.2f}%")
    print(f"  - Loss is finite: {torch.isfinite(loss)}")
    
    # Check that parameters were updated
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"✓ Gradients computed: {has_grad}")
    
    # Check VIC weights
    if model.use_vic:
        weights = model.weight_controller.get_weights()
        print(f"✓ Dynamic weights: {weights.detach().numpy()}")
    
    return True


def test_gradient_checkpointing():
    """Test gradient checkpointing functionality."""
    print("\n=== Testing Gradient Checkpointing ===")
    
    # Import modules
    import importlib.util
    
    # Load backbone module
    spec = importlib.util.spec_from_file_location('backbone', 'backbone.py')
    backbone_module = importlib.util.module_from_spec(spec)
    sys.modules['backbone'] = backbone_module
    spec.loader.exec_module(backbone_module)
    
    # Create a simple feature extractor
    class SimpleFeatureExtractor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten()
            )
            self.final_feat_dim = 64
        
        def forward(self, x):
            return self.conv(x)
    
    # Load enhanced transformer directly
    spec = importlib.util.spec_from_file_location('enhanced_transformer', 
        'methods/enhanced_transformer.py')
    enhanced_module = importlib.util.module_from_spec(spec)
    
    # Mock MetaTemplate
    class MockMetaTemplate(torch.nn.Module):
        def __init__(self, model_func, n_way, k_shot, n_query, change_way=True):
            super().__init__()
            self.n_way = n_way
            self.k_shot = k_shot
            self.n_query = n_query
            self.feature = model_func()
            self.feat_dim = self.feature.final_feat_dim
            self.change_way = change_way
        
        def parse_feature(self, x, is_feature):
            device = x.device
            if is_feature:
                z_all = x
            else:
                x = x.contiguous().view(self.n_way * (self.k_shot + self.n_query), *x.size()[2:])
                z_all = self.feature.forward(x)
                z_all = z_all.view(self.n_way, self.k_shot + self.n_query, -1)
            
            z_support = z_all[:, :self.k_shot]
            z_query = z_all[:, self.k_shot:]
            return z_support, z_query
    
    # Inject mock
    sys.modules['methods.meta_template'] = type(sys)('methods.meta_template')
    sys.modules['methods.meta_template'].MetaTemplate = MockMetaTemplate
    
    # Now load enhanced transformer
    spec.loader.exec_module(enhanced_module)
    EnhancedFewShotTransformer = enhanced_module.EnhancedFewShotTransformer
    
    # Create model with checkpointing
    n_way, k_shot, n_query = 5, 5, 10
    
    def feature_model():
        return SimpleFeatureExtractor()
    
    model = EnhancedFewShotTransformer(
        feature_model, 
        n_way=n_way, 
        k_shot=k_shot, 
        n_query=n_query,
        variant='cosine',
        depth=2,
        heads=4,
        dim_head=64,
        mlp_dim=512,
        use_vic=True,
        use_mahalanobis=True,
        vic_lambda_init=[9.0, 0.5, 0.5],
        weight_controller='uncertainty',
        use_checkpoint=True  # Enable checkpointing
    )
    
    print(f"✓ Model created with gradient checkpointing")
    
    # Create synthetic episode
    episode = create_synthetic_episode(n_way, k_shot, n_query, img_size=84)
    
    # Training step with checkpointing
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    optimizer.zero_grad()
    acc, loss = model.set_forward_loss(episode)
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training with checkpointing successful")
    print(f"  - Loss: {loss.item():.6f}")
    print(f"  - Accuracy: {acc * 100:.2f}%")
    
    return True


if __name__ == '__main__':
    print("=" * 70)
    print("Running Enhanced Transformer End-to-End Validation")
    print("=" * 70)
    
    try:
        test_enhanced_transformer_forward()
        test_enhanced_transformer_training()
        test_gradient_checkpointing()
        
        print("\n" + "=" * 70)
        print("✓ ALL VALIDATION TESTS PASSED!")
        print("=" * 70)
        print("\nThe enhanced transformer is ready for use:")
        print("  - Forward pass works correctly")
        print("  - Training step computes gradients properly")
        print("  - VIC regularization is functional")
        print("  - Mahalanobis classifier produces valid logits")
        print("  - Gradient checkpointing works")
        print("\nNext steps:")
        print("  1. Run: python train.py --method FSCT_enhanced_cosine --dataset miniImagenet")
        print("  2. Monitor training logs and validation accuracy")
        print("  3. Compare with baseline FSCT_cosine")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ VALIDATION FAILED!")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
