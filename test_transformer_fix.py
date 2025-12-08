"""
Test for FewShotTransformer LayerNorm dimension fix

This test verifies that FewShotTransformer works correctly by using a test
forward pass to dynamically determine the actual feature dimension, rather than
relying on the backbone's reported final_feat_dim which may be incorrect.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_transformer_with_flatten_false():
    """Test FewShotTransformer with flatten=False"""
    print("Testing FewShotTransformer with flatten=False")
    print("=" * 80)
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("✗ PyTorch not available, cannot run test")
        return False
    
    try:
        from methods.transformer import FewShotTransformer
        from io_utils import model_dict
        print("✓ Modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test parameters
    n_way = 5
    k_shot = 1
    n_query = 16
    dataset = 'miniImagenet'
    backbones_to_test = ['Conv4', 'Conv6']
    
    # Test with ResNet if available
    try:
        backbones_to_test.append('ResNet18')
    except:
        pass
    
    for backbone_name in backbones_to_test:
        if backbone_name not in model_dict:
            print(f"⚠ Skipping {backbone_name} - not in model_dict")
            continue
            
        print(f"\nTesting with {backbone_name}...")
        
        try:
            model_func = model_dict[backbone_name]
            
            # Create model with flatten=False
            model = FewShotTransformer(
                model_func,
                n_way=n_way,
                k_shot=k_shot,
                n_query=n_query,
                variant='cosine',
                depth=1,
                heads=4,
                dim_head=64,
                mlp_dim=512,
                dataset=dataset,
                feti=0,
                flatten=False
            )
            
            print(f"  ✓ Model created successfully")
            print(f"  - feat_dim: {model.feat_dim}")
            
            # Test forward pass with dummy data
            model.eval()
            with torch.no_grad():
                # Create dummy episode: [n_way, k_shot + n_query, C, H, W]
                # miniImagenet uses 84x84 images with 3 channels
                dummy_data = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
                
                try:
                    # Test set_forward
                    scores = model.set_forward(dummy_data, is_feature=False)
                    print(f"  ✓ Forward pass successful")
                    print(f"  - Output shape: {scores.shape}")
                    print(f"  - Expected shape: ({n_way * n_query}, {n_way})")
                    
                    # Verify output shape is correct
                    expected_shape = (n_way * n_query, n_way)
                    if scores.shape == expected_shape:
                        print(f"  ✓ Output shape is correct")
                    else:
                        print(f"  ✗ Output shape mismatch: got {scores.shape}, expected {expected_shape}")
                        return False
                    
                    # Test set_forward_loss
                    acc, loss = model.set_forward_loss(dummy_data)
                    print(f"  ✓ Loss computation successful")
                    print(f"  - Accuracy: {acc:.4f}")
                    print(f"  - Loss: {loss.item():.4f}")
                    
                except Exception as e:
                    print(f"  ✗ Forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
                    
        except Exception as e:
            print(f"  ✗ Model creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    return True


def test_transformer_with_flatten_true():
    """Test FewShotTransformer with flatten=True (old behavior)"""
    print("\nTesting FewShotTransformer with flatten=True (for backward compatibility)")
    print("=" * 80)
    
    try:
        import torch
        from methods.transformer import FewShotTransformer
        from io_utils import model_dict
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    n_way = 5
    k_shot = 1
    n_query = 16
    dataset = 'miniImagenet'
    backbone_name = 'Conv4'
    
    try:
        model_func = model_dict[backbone_name]
        
        # Create model with flatten=True
        model = FewShotTransformer(
            model_func,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            variant='cosine',
            depth=1,
            heads=4,
            dim_head=64,
            mlp_dim=512,
            dataset=dataset,
            feti=0,
            flatten=True
        )
        
        print(f"✓ Model created successfully with flatten=True")
        print(f"  - feat_dim: {model.feat_dim}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            dummy_data = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
            
            try:
                scores = model.set_forward(dummy_data, is_feature=False)
                print(f"✓ Forward pass successful with flatten=True")
                print(f"  - Output shape: {scores.shape}")
            except Exception as e:
                print(f"✗ Forward pass failed: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Backward compatibility maintained!")
    return True


if __name__ == '__main__':
    success = True
    
    # Test with flatten=False (the fix)
    if not test_transformer_with_flatten_false():
        success = False
    
    # Test with flatten=True (backward compatibility)
    if not test_transformer_with_flatten_true():
        success = False
    
    if success:
        print("\n" + "=" * 80)
        print("All tests passed successfully! ✓")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("Some tests failed! ✗")
        sys.exit(1)
