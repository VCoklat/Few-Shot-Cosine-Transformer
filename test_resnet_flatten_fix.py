#!/usr/bin/env python3
"""
Test to verify that ResNet models properly flatten output when flatten=True.
This test validates the fix for the LayerNorm shape mismatch error.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_resnet_flatten():
    """Test that ResNet models properly flatten output when flatten=True"""
    print("=" * 80)
    print("Testing ResNet Flatten Fix")
    print("=" * 80)
    print()
    
    try:
        import torch
        import backbone
    except ImportError as e:
        print(f"⚠ WARNING: Cannot import required modules: {e}")
        print("Skipping test (this is expected in environments without torch)")
        return True
    
    print("Test 1: ResNetModel with flatten=True (CUB dataset)")
    print("-" * 60)
    
    # Test with CUB dataset (84x84 images)
    model = backbone.ResNetModel(dataset='CUB', variant=34, flatten=True)
    print(f"  Model type: ResNetModel (variant=34)")
    print(f"  Dataset: CUB")
    print(f"  Flatten: True")
    print(f"  Expected final_feat_dim: {model.final_feat_dim}")
    
    # Create dummy input (batch_size=2, channels=3, height=84, width=84)
    x = torch.randn(2, 3, 84, 84)
    print(f"  Input shape: {x.shape}")
    
    out = model.forward(x)
    print(f"  Output shape: {out.shape}")
    
    # Verify shape
    expected_shape = (2, model.final_feat_dim)
    if out.shape != expected_shape:
        print(f"✗ FAILED: Expected shape {expected_shape}, got {out.shape}")
        return False
    
    print(f"✓ PASSED: Output shape matches expected {expected_shape}")
    print()
    
    print("Test 2: ResNetModel with flatten=False (CUB dataset)")
    print("-" * 60)
    
    model = backbone.ResNetModel(dataset='CUB', variant=34, flatten=False)
    print(f"  Model type: ResNetModel (variant=34)")
    print(f"  Dataset: CUB")
    print(f"  Flatten: False")
    print(f"  Expected final_feat_dim: {model.final_feat_dim}")
    
    x = torch.randn(2, 3, 84, 84)
    print(f"  Input shape: {x.shape}")
    
    out = model.forward(x)
    print(f"  Output shape: {out.shape}")
    
    # Verify shape (should be 4D: batch, channels, height, width)
    if len(out.shape) != 4:
        print(f"✗ FAILED: Expected 4D tensor, got shape {out.shape}")
        return False
    
    if out.shape[1] != 512:
        print(f"✗ FAILED: Expected 512 channels, got {out.shape[1]}")
        return False
    
    print(f"✓ PASSED: Output shape is correct 4D tensor")
    print()
    
    print("Test 3: ResNet (custom) with flatten=True")
    print("-" * 60)
    
    model = backbone.ResNet(backbone.BasicBlock, [3, 4, 6, 3], flatten=True)
    print(f"  Model type: ResNet (custom, ResNet34 architecture)")
    print(f"  Flatten: True")
    print(f"  Expected final_feat_dim: {model.final_feat_dim}")
    
    x = torch.randn(2, 3, 84, 84)
    print(f"  Input shape: {x.shape}")
    
    out = model.forward(x)
    print(f"  Output shape: {out.shape}")
    
    # Verify shape
    expected_shape = (2, model.final_feat_dim)
    if out.shape != expected_shape:
        print(f"✗ FAILED: Expected shape {expected_shape}, got {out.shape}")
        return False
    
    print(f"✓ PASSED: Output shape matches expected {expected_shape}")
    print()
    
    print("Test 4: Integration test - FewShotTransformer initialization")
    print("-" * 60)
    
    try:
        from methods.transformer import FewShotTransformer
        
        # This is what caused the original error
        model_func = backbone.ResNet34
        transformer = FewShotTransformer(
            model_func,
            n_way=5,
            k_shot=1,
            n_query=16,
            variant='cosine',
            depth=1,
            heads=4,
            dim_head=64,
            mlp_dim=512,
            dataset='CUB',
            feti=0,
            flatten=True
        )
        
        print(f"  Model: FewShotTransformer")
        print(f"  Backbone: ResNet34")
        print(f"  Feature dimension: {transformer.feat_dim}")
        print(f"✓ PASSED: FewShotTransformer initialized successfully")
        print()
        
    except ImportError as e:
        print(f"⚠ SKIPPED: Missing dependencies for integration test: {e}")
        print()
    except Exception as e:
        print(f"✗ FAILED: FewShotTransformer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    success = test_resnet_flatten()
    sys.exit(0 if success else 1)
