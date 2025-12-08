#!/usr/bin/env python3
"""
Test script to verify the fixes for HAM10000 and Omniglot issues.
This script validates:
1. HAM10000 dataset has correct class splits
2. ConvNet calculates correct feature dimensions for different datasets
"""

import json
import sys
import os

def test_ham10000_splits():
    """Test that HAM10000 dataset has correct number of classes in each split"""
    print("="*60)
    print("TEST 1: HAM10000 Dataset Splits")
    print("="*60)
    
    base_path = "dataset/HAM10000/base.json"
    val_path = "dataset/HAM10000/val.json"
    novel_path = "dataset/HAM10000/novel.json"
    
    # Check base.json
    with open(base_path, 'r') as f:
        base_data = json.load(f)
    base_classes = len(set(base_data['image_labels']))
    print(f"\nBase split:")
    print(f"  Classes: {base_classes}")
    print(f"  Expected: >= 3 (for 3-way classification)")
    
    if base_classes >= 3:
        print("  ✅ PASS: Base split has enough classes for 3-way classification")
    else:
        print(f"  ❌ FAIL: Base split has only {base_classes} classes")
        return False
    
    # Check val.json
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    val_classes = len(set(val_data['image_labels']))
    print(f"\nValidation split:")
    print(f"  Classes: {val_classes}")
    print(f"  Expected: >= 2 (for 2-way classification)")
    
    if val_classes >= 2:
        print("  ✅ PASS: Validation split has enough classes")
    else:
        print(f"  ❌ FAIL: Validation split has only {val_classes} classes")
        return False
    
    # Check novel.json
    with open(novel_path, 'r') as f:
        novel_data = json.load(f)
    novel_classes = len(set(novel_data['image_labels']))
    print(f"\nNovel split:")
    print(f"  Classes: {novel_classes}")
    print(f"  Expected: >= 2 (for 2-way classification)")
    
    if novel_classes >= 2:
        print("  ✅ PASS: Novel split has enough classes")
    else:
        print(f"  ❌ FAIL: Novel split has only {novel_classes} classes")
        return False
    
    return True


def test_convnet_dimensions():
    """Test that ConvNet calculates correct feature dimensions"""
    print("\n" + "="*60)
    print("TEST 2: ConvNet Feature Dimensions")
    print("="*60)
    
    # Test cases: (dataset, input_size, depth, expected_feat_dim)
    test_cases = [
        ('Omniglot', 28, 4, 64),      # 28 / 16 = 1, 64*1*1 = 64
        ('cross_char', 28, 4, 64),    # Same as Omniglot
        ('CIFAR', 32, 4, 256),        # 32 / 16 = 2, 64*2*2 = 256
        ('miniImagenet', 84, 4, 1600),# 84 / 16 = 5, 64*5*5 = 1600
        ('CUB', 84, 4, 1600),         # Same as miniImagenet
    ]
    
    all_passed = True
    
    for dataset, input_size, depth, expected_dim in test_cases:
        num_pools = min(depth, 4)
        spatial_dim = input_size // (2 ** num_pools)
        calculated_dim = 64 * spatial_dim * spatial_dim
        
        print(f"\n{dataset}:")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Depth: {depth} (pooling layers: {num_pools})")
        print(f"  Spatial output: {spatial_dim}x{spatial_dim}")
        print(f"  Feature dimension: {calculated_dim}")
        print(f"  Expected: {expected_dim}")
        
        if calculated_dim == expected_dim:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL: Got {calculated_dim}, expected {expected_dim}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TESTING FIXES FOR EXPERIMENT ERRORS")
    print("="*60 + "\n")
    
    # Run tests
    test1_passed = test_ham10000_splits()
    test2_passed = test_convnet_dimensions()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"HAM10000 Splits: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"ConvNet Dimensions: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All tests PASSED!")
        print("\nThe fixes should resolve:")
        print("  1. HAM10000 3-way classification error")
        print("  2. Omniglot Conv4 shape mismatch error")
        return 0
    else:
        print("\n❌ Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
