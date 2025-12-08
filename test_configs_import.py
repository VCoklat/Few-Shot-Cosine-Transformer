#!/usr/bin/env python3
"""
Test to verify that configs module properly exports data_dir and save_dir.
This test ensures the fix for AttributeError: module 'configs' has no attribute 'data_dir'
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_configs_has_data_dir():
    """Test that configs module has data_dir attribute"""
    print("Testing: configs module has data_dir...")
    import configs
    
    assert hasattr(configs, 'data_dir'), "configs module should have 'data_dir' attribute"
    assert isinstance(configs.data_dir, dict), "data_dir should be a dictionary"
    print("  ✓ configs.data_dir exists and is a dictionary")


def test_configs_has_save_dir():
    """Test that configs module has save_dir attribute"""
    print("Testing: configs module has save_dir...")
    import configs
    
    assert hasattr(configs, 'save_dir'), "configs module should have 'save_dir' attribute"
    print("  ✓ configs.save_dir exists")


def test_configs_data_dir_contains_datasets():
    """Test that data_dir contains expected dataset keys"""
    print("Testing: data_dir contains expected datasets...")
    import configs
    
    expected_datasets = ['CUB', 'miniImagenet', 'Omniglot', 'emnist', 
                        'Yoga', 'CIFAR', 'HAM10000', 'DatasetIndo']
    
    for dataset in expected_datasets:
        assert dataset in configs.data_dir, f"data_dir should contain '{dataset}'"
    
    print(f"  ✓ data_dir contains all {len(expected_datasets)} expected datasets")


def test_configs_import_as_data_configs():
    """Test that importing configs as data_configs works (mimics run_experiments.py)"""
    print("Testing: import configs as data_configs...")
    import configs as data_configs
    
    assert hasattr(data_configs, 'data_dir'), "data_configs should have 'data_dir' attribute"
    assert isinstance(data_configs.data_dir, dict), "data_dir should be a dictionary"
    
    # Test the exact pattern from run_experiments.py line 220
    test_dataset = 'miniImagenet'
    assert test_dataset in data_configs.data_dir, f"data_configs.data_dir should contain '{test_dataset}'"
    
    print("  ✓ import configs as data_configs works correctly")


def test_backward_compatibility():
    """Test backward compatibility with existing code patterns"""
    print("Testing: backward compatibility with existing code...")
    import configs
    
    # Test pattern from train.py: configs.data_dir['miniImagenet']
    try:
        path = configs.data_dir['miniImagenet']
        assert isinstance(path, str), "Dataset path should be a string"
        print(f"  ✓ configs.data_dir['miniImagenet'] = {path}")
    except KeyError:
        raise AssertionError("Failed to access configs.data_dir['miniImagenet']")
    
    # Test pattern from train.py: configs.data_dir['CUB']
    try:
        path = configs.data_dir['CUB']
        assert isinstance(path, str), "Dataset path should be a string"
        print(f"  ✓ configs.data_dir['CUB'] = {path}")
    except KeyError:
        raise AssertionError("Failed to access configs.data_dir['CUB']")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Testing configs module data_dir and save_dir export")
    print("="*60)
    print()
    
    try:
        test_configs_has_data_dir()
        test_configs_has_save_dir()
        test_configs_data_dir_contains_datasets()
        test_configs_import_as_data_configs()
        test_backward_compatibility()
        
        print()
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        return True
    
    except AssertionError as e:
        print()
        print("="*60)
        print(f"✗ TEST FAILED: {e}")
        print("="*60)
        return False
    
    except Exception as e:
        print()
        print("="*60)
        print(f"✗ UNEXPECTED ERROR: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
