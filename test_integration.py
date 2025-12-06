"""
Integration Test for Enhanced Few-Shot Learning

Tests the integration of all components without requiring datasets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_integration():
    """Test complete integration of enhanced model"""
    print("Testing Enhanced Model Integration")
    print("=" * 80)
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("✗ PyTorch not available, cannot run integration test")
        return False
    
    try:
        # Import all components
        from models.optimal_fewshot_enhanced import get_model_for_dataset, EnhancedOptimalFewShot
        from modules.task_invariance import TaskAdaptiveInvariance, MultiScaleInvariance
        from modules.feature_augmentation import FeatureLevelAugmentation, PrototypicalRefinement
        from modules.medical_invariance import MedicalImageInvariance, ContrastiveInvarianceLoss
        
        print("✓ All modules imported successfully")
        
        # Test model creation for each dataset
        datasets = ['omniglot', 'miniimagenet', 'ham10000', 'cub']
        
        for dataset in datasets:
            print(f"\nTesting {dataset}...")
            
            model = get_model_for_dataset(
                dataset=dataset,
                model_func=lambda: None,
                n_way=5,
                k_shot=1,
                n_query=16
            )
            
            # Check model configuration
            print(f"  - Task invariance: {model.use_task_invariance}")
            print(f"  - Multi-scale: {model.use_multi_scale}")
            print(f"  - Feature aug: {model.use_feature_augmentation}")
            print(f"  - Proto refine: {model.use_prototype_refinement}")
            print(f"  - Domain: {model.domain}")
            
            # Test forward pass with dummy data
            model.eval()
            with torch.no_grad():
                # Create dummy episode: [n_way, k_shot + n_query, C, H, W]
                dummy_data = torch.randn(5, 17, 3, 84, 84)
                
                try:
                    logits = model.set_forward(dummy_data)
                    expected_shape = (80, 5)  # 5-way * 16 query, 5 classes
                    assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} vs {expected_shape}"
                    print(f"  ✓ Forward pass successful, output shape: {logits.shape}")
                except Exception as e:
                    print(f"  ✗ Forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
        
        print("\n" + "=" * 80)
        print("✓ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation"""
    print("\n" + "=" * 80)
    print("Testing Loss Computation")
    print("=" * 80)
    
    try:
        import torch
        from models.optimal_fewshot_enhanced import get_model_for_dataset
        
        model = get_model_for_dataset(
            dataset='miniimagenet',
            model_func=lambda: None,
            n_way=5,
            k_shot=1,
            n_query=16
        )
        
        model.train()  # Training mode
        
        # Create dummy episode
        dummy_data = torch.randn(5, 17, 3, 84, 84)
        
        # Compute loss
        acc, loss = model.set_forward_loss(dummy_data)
        
        print(f"✓ Loss computation successful")
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - Loss: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_medical_domain_specific():
    """Test medical domain-specific components"""
    print("\n" + "=" * 80)
    print("Testing Medical Domain-Specific Components")
    print("=" * 80)
    
    try:
        import torch
        from models.optimal_fewshot_enhanced import get_model_for_dataset
        
        model = get_model_for_dataset(
            dataset='ham10000',
            model_func=lambda: None,
            n_way=7,  # HAM10000 specific
            k_shot=5,
            n_query=16
        )
        
        # Check medical-specific components
        assert model.domain == 'medical', "Domain should be 'medical'"
        assert hasattr(model, 'medical_invariance'), "Should have medical_invariance module"
        assert hasattr(model, 'contrastive_loss'), "Should have contrastive_loss"
        
        print("✓ Medical-specific components present")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            dummy_data = torch.randn(7, 21, 3, 84, 84)  # 7-way, 5-shot + 16 query
            logits = model.set_forward(dummy_data)
            expected_shape = (112, 7)  # 7-way * 16 query, 7 classes
            assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape}"
            print(f"✓ Forward pass successful for 7-way task")
        
        return True
        
    except Exception as e:
        print(f"✗ Medical domain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("=" * 80)
    print("Enhanced Few-Shot Learning - Integration Tests")
    print("=" * 80)
    print()
    
    results = []
    
    # Run tests
    results.append(("Integration", test_integration()))
    results.append(("Loss Computation", test_loss_computation()))
    results.append(("Medical Domain", test_medical_domain_specific()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("✓ All integration tests passed!")
        return 0
    else:
        print("✗ Some integration tests failed")
        return 1


if __name__ == '__main__':
    exit(main())
