"""
Test script to verify that OptimalFewShotModel works with different backbones
"""
import torch
import numpy as np
import backbone
from methods.optimal_few_shot import OptimalFewShotModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_backbone(backbone_name, dataset='miniImagenet', image_size=84):
    """Test OptimalFewShotModel with a specific backbone"""
    print(f"\n{'='*60}")
    print(f"Testing backbone: {backbone_name}")
    print(f"Dataset: {dataset}, Image size: {image_size}")
    print(f"{'='*60}")
    
    # Create model function based on backbone
    if backbone_name == 'Conv4':
        def feature_model():
            return backbone.Conv4(dataset, flatten=True)
        expected_feat_dim = 1600  # 64 * 5 * 5
    elif backbone_name == 'ResNet34':
        def feature_model():
            return backbone.ResNet34(False, dataset, flatten=True)
        expected_feat_dim = 25088  # 512 * 7 * 7
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    # Create model
    try:
        model = OptimalFewShotModel(
            feature_model,
            n_way=5,
            k_shot=1,
            n_query=15,
            feature_dim=64,
            n_heads=4,
            dropout=0.1,
            num_datasets=5,
            dataset=dataset,
            use_focal_loss=False,
            label_smoothing=0.1
        )
        model = model.to(device)
        print(f"✓ Model created successfully")
        print(f"  Feature dimension: {model.feat_dim}")
        print(f"  Expected dimension: {expected_feat_dim}")
        assert model.feat_dim == expected_feat_dim, f"Feature dimension mismatch: {model.feat_dim} != {expected_feat_dim}"
        
        # Test forward pass
        batch_size = 5 * (1 + 15)  # n_way * (k_shot + n_query)
        x = torch.randn(5, batch_size, 3, image_size, image_size).to(device)
        
        print(f"\n  Testing forward pass with input shape: {x.shape}")
        
        # Test parse_feature
        z_support, z_query = model.parse_feature(x, is_feature=False)
        print(f"  Support features shape: {z_support.shape}")
        print(f"  Query features shape: {z_query.shape}")
        
        # Expected shapes
        # z_support: [n_way, k_shot, feat_dim]
        # z_query: [n_way, n_query, feat_dim]
        assert z_support.shape == (5, 1, model.feat_dim), f"Support shape mismatch: {z_support.shape}"
        assert z_query.shape == (5, 15, model.feat_dim), f"Query shape mismatch: {z_query.shape}"
        
        print(f"✓ Forward pass successful")
        
        # Test full forward with projection
        print(f"\n  Testing projection layer...")
        support_flat = z_support.reshape(-1, model.feat_dim)
        query_flat = z_query.reshape(-1, model.feat_dim)
        
        support_projected = model.projection(support_flat)
        query_projected = model.projection(query_flat)
        
        print(f"  Projected support shape: {support_projected.shape}")
        print(f"  Projected query shape: {query_projected.shape}")
        
        assert support_projected.shape == (5, 64), f"Projected support shape mismatch: {support_projected.shape}"
        assert query_projected.shape == (75, 64), f"Projected query shape mismatch: {query_projected.shape}"
        
        print(f"✓ Projection successful")
        print(f"\n✓✓✓ All tests passed for {backbone_name} ✓✓✓")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING BACKBONE COMPATIBILITY FIX")
    print("="*60)
    
    results = {}
    
    # Test Conv4 (original backbone)
    results['Conv4'] = test_backbone('Conv4', dataset='miniImagenet', image_size=84)
    
    # Test ResNet34 (the problematic backbone)
    results['ResNet34'] = test_backbone('ResNet34', dataset='miniImagenet', image_size=224)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for backbone, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{backbone:20s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
    
    exit(0 if all_passed else 1)
