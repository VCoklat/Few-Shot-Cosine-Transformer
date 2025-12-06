#!/usr/bin/env python3
"""
Basic validation script for enhanced few-shot learning modules.

This script validates that all modules can be imported and instantiated
without errors. It does not require a GPU or dataset.
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from modules.task_invariance import TaskAdaptiveInvariance, MultiScaleInvariance
        print("✓ task_invariance module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import task_invariance: {e}")
        return False
    
    try:
        from modules.feature_augmentation import FeatureLevelAugmentation, PrototypicalRefinement
        print("✓ feature_augmentation module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import feature_augmentation: {e}")
        return False
    
    try:
        from modules.medical_invariance import MedicalImageInvariance, ContrastiveInvarianceLoss
        print("✓ medical_invariance module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import medical_invariance: {e}")
        return False
    
    try:
        from models.optimal_fewshot_enhanced import EnhancedOptimalFewShot, get_model_for_dataset
        print("✓ optimal_fewshot_enhanced module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import optimal_fewshot_enhanced: {e}")
        return False
    
    return True


def test_module_instantiation():
    """Test that modules can be instantiated"""
    print("\nTesting module instantiation...")
    
    try:
        # Only test if torch is available
        import torch
    except ImportError:
        print("⚠ PyTorch not available, skipping instantiation tests")
        return True
    
    from modules.task_invariance import TaskAdaptiveInvariance, MultiScaleInvariance
    from modules.feature_augmentation import FeatureLevelAugmentation, PrototypicalRefinement
    from modules.medical_invariance import MedicalImageInvariance, ContrastiveInvarianceLoss
    
    feature_dim = 64
    
    try:
        # Test TaskAdaptiveInvariance
        module = TaskAdaptiveInvariance(feature_dim=feature_dim)
        print("✓ TaskAdaptiveInvariance instantiated")
        
        # Test MultiScaleInvariance
        module = MultiScaleInvariance(feature_dim=feature_dim)
        print("✓ MultiScaleInvariance instantiated")
        
        # Test FeatureLevelAugmentation
        module = FeatureLevelAugmentation(feature_dim=feature_dim)
        print("✓ FeatureLevelAugmentation instantiated")
        
        # Test PrototypicalRefinement
        module = PrototypicalRefinement(feature_dim=feature_dim)
        print("✓ PrototypicalRefinement instantiated")
        
        # Test MedicalImageInvariance
        module = MedicalImageInvariance(feature_dim=feature_dim)
        print("✓ MedicalImageInvariance instantiated")
        
        # Test ContrastiveInvarianceLoss
        loss_fn = ContrastiveInvarianceLoss()
        print("✓ ContrastiveInvarianceLoss instantiated")
        
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate modules: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward passes with dummy data"""
    print("\nTesting forward passes...")
    
    try:
        import torch
    except ImportError:
        print("⚠ PyTorch not available, skipping forward pass tests")
        return True
    
    from modules.task_invariance import TaskAdaptiveInvariance, MultiScaleInvariance
    from modules.feature_augmentation import FeatureLevelAugmentation, PrototypicalRefinement
    from modules.medical_invariance import MedicalImageInvariance, ContrastiveInvarianceLoss
    
    feature_dim = 64
    batch_size = 10
    
    try:
        # Create dummy input
        x = torch.randn(batch_size, feature_dim)
        
        # Test TaskAdaptiveInvariance
        module = TaskAdaptiveInvariance(feature_dim=feature_dim)
        output = module(x)
        assert output.shape == (batch_size, feature_dim), "TaskAdaptiveInvariance output shape mismatch"
        print("✓ TaskAdaptiveInvariance forward pass successful")
        
        # Test MultiScaleInvariance
        module = MultiScaleInvariance(feature_dim=feature_dim)
        output = module(x)
        assert output.shape == (batch_size, feature_dim), "MultiScaleInvariance output shape mismatch"
        print("✓ MultiScaleInvariance forward pass successful")
        
        # Test FeatureLevelAugmentation
        module = FeatureLevelAugmentation(feature_dim=feature_dim)
        output = module(x, is_training=True)
        assert output.shape == (batch_size, feature_dim), "FeatureLevelAugmentation output shape mismatch"
        print("✓ FeatureLevelAugmentation forward pass successful")
        
        # Test PrototypicalRefinement
        module = PrototypicalRefinement(feature_dim=feature_dim)
        prototypes = torch.randn(5, feature_dim)  # 5-way
        query_features = torch.randn(80, feature_dim)  # 5-way * 16 queries
        output = module(prototypes, query_features)
        assert output.shape == (5, feature_dim), "PrototypicalRefinement output shape mismatch"
        print("✓ PrototypicalRefinement forward pass successful")
        
        # Test MedicalImageInvariance
        module = MedicalImageInvariance(feature_dim=feature_dim)
        output = module(x)
        assert output.shape == (batch_size, feature_dim), "MedicalImageInvariance output shape mismatch"
        print("✓ MedicalImageInvariance forward pass successful")
        
        # Test ContrastiveInvarianceLoss
        loss_fn = ContrastiveInvarianceLoss()
        x2 = torch.randn(batch_size, feature_dim)
        loss = loss_fn(x, x2)
        assert loss.dim() == 0, "ContrastiveInvarianceLoss should return scalar"
        print("✓ ContrastiveInvarianceLoss forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        traceback.print_exc()
        return False


def test_factory_function():
    """Test the factory function for creating models"""
    print("\nTesting factory function...")
    
    try:
        import torch
    except ImportError:
        print("⚠ PyTorch not available, skipping factory function test")
        return True
    
    try:
        from models.optimal_fewshot_enhanced import get_model_for_dataset
        
        # Test with different datasets
        datasets = ['omniglot', 'miniimagenet', 'ham10000', 'cub']
        
        for dataset in datasets:
            # Create a simple model_func that returns None (will use default Conv4)
            model_func = lambda: None
            
            model = get_model_for_dataset(
                dataset=dataset,
                model_func=model_func,
                n_way=5,
                k_shot=1,
                n_query=16
            )
            
            print(f"✓ Factory function created model for {dataset}")
        
        return True
    except Exception as e:
        print(f"✗ Factory function failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("=" * 80)
    print("Enhanced Few-Shot Learning Module Validation")
    print("=" * 80)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test instantiation
    results.append(("Instantiation", test_module_instantiation()))
    
    # Test forward passes
    results.append(("Forward Passes", test_forward_pass()))
    
    # Test factory function
    results.append(("Factory Function", test_factory_function()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("✓ All validation tests passed!")
        return 0
    else:
        print("✗ Some validation tests failed")
        return 1


if __name__ == '__main__':
    exit(main())
