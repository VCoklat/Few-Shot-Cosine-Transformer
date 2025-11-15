#!/usr/bin/env python3
"""
Test script to verify the correct() method fix for OptimalFewShotModel
This specifically tests that the correct() method handles the tuple return from set_forward()
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.optimal_few_shot import OptimalFewShotModel

def test_correct_method():
    """Test that correct() method works properly with tuple return from set_forward()"""
    print("Testing correct() method fix...")
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def dummy_model_func():
        return None
    
    model = OptimalFewShotModel(
        dummy_model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        feature_dim=64,
        n_heads=4,
        dropout=0.1,
        num_datasets=5,
        dataset='miniImagenet',
        use_focal_loss=False,
        label_smoothing=0.1
    )
    
    # Create dummy episode data
    # Shape: (n_way, k_shot + n_query, C, H, W)
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    print(f"  Input shape: {x.shape}")
    
    # Test correct method - this should not raise an AttributeError
    model.eval()
    with torch.no_grad():
        try:
            correct_this, count_this = model.correct(x)
            print(f"  Correct predictions: {correct_this}/{count_this}")
            print(f"  Accuracy: {correct_this/count_this*100:.2f}%")
            print("✓ correct() method test passed - no AttributeError!")
            return True
        except AttributeError as e:
            print(f"✗ correct() method test failed with AttributeError: {e}")
            return False
        except Exception as e:
            print(f"✗ correct() method test failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_validation_loop_simulation():
    """Simulate a validation loop to ensure it works like val_loop"""
    print("\nTesting validation loop simulation...")
    
    n_way = 5
    k_shot = 5
    n_query = 15
    
    def dummy_model_func():
        return None
    
    model = OptimalFewShotModel(
        dummy_model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        feature_dim=64,
        n_heads=4,
        dropout=0.1,
        num_datasets=5,
        dataset='miniImagenet',
        use_focal_loss=False,
        label_smoothing=0.1
    )
    
    # Simulate a few validation episodes
    model.eval()
    acc_all = []
    
    with torch.no_grad():
        for i in range(5):
            x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
            correct_this, count_this = model.correct(x)
            acc = correct_this / count_this * 100
            acc_all.append(acc)
            print(f"  Episode {i+1}: {acc:.2f}% accuracy")
    
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    
    print(f"  Average validation accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
    print("✓ Validation loop simulation test passed!")
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("Testing correct() method fix for OptimalFewShotModel")
    print("="*60)
    
    try:
        success1 = test_correct_method()
        success2 = test_validation_loop_simulation()
        
        if success1 and success2:
            print("\n" + "="*60)
            print("✓ All tests passed successfully!")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("✗ Some tests failed!")
            print("="*60)
            return 1
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ Test failed with error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
