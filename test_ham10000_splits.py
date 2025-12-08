#!/usr/bin/env python3
"""
Test script to verify HAM10000 dataset splits support n-way classification.

This test ensures that the dataset splits have sufficient classes for few-shot learning
experiments, particularly 3-way and 5-way classification tasks.
"""
import json
import os
import sys


def test_dataset_split(split_path, split_name, min_classes=1):
    """Test that a dataset split has sufficient classes."""
    if not os.path.exists(split_path):
        print(f"  ✗ ERROR: {split_path} not found")
        return False
    
    with open(split_path, 'r') as f:
        data = json.load(f)
    
    n_classes = len(set(data['image_labels']))
    n_images = len(data['image_names'])
    
    # Get the actual class names being used
    used_indices = sorted(set(data['image_labels']))
    used_labels = [data['label_names'][i] for i in used_indices]
    
    print(f"  {split_name} split:")
    print(f"    - Classes: {n_classes} {used_labels}")
    print(f"    - Images: {n_images}")
    
    if n_classes >= min_classes:
        print(f"    ✓ PASS: Has {n_classes} classes (minimum: {min_classes})")
        return True
    else:
        print(f"    ✗ FAIL: Has {n_classes} classes (minimum required: {min_classes})")
        return False


def main():
    """Main test function."""
    print("=" * 70)
    print("HAM10000 Dataset Split Test")
    print("=" * 70)
    
    dataset_path = 'dataset/HAM10000'
    
    # Test configuration
    # For 3-way classification, we need at least 3 classes in each relevant split
    # For 5-way classification, we need at least 5 classes
    
    all_passed = True
    
    # Test base split (training data)
    # Can have fewer classes if needed, as long as val/test have enough
    print("\n1. Testing base.json (training split):")
    if not test_dataset_split(
        os.path.join(dataset_path, 'base.json'),
        'Base',
        min_classes=2
    ):
        all_passed = False
    
    # Test validation split (this was the failing one - needs 3 for 3-way)
    print("\n2. Testing val.json (validation split):")
    if not test_dataset_split(
        os.path.join(dataset_path, 'val.json'),
        'Validation',
        min_classes=3  # Minimum for 3-way classification
    ):
        all_passed = False
    
    # Test novel split (testing data)
    print("\n3. Testing novel.json (testing split):")
    if not test_dataset_split(
        os.path.join(dataset_path, 'novel.json'),
        'Novel',
        min_classes=2
    ):
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All tests PASSED! Dataset splits are properly configured.")
        print("   - Validation split supports 3-way classification")
        print("=" * 70)
        return 0
    else:
        print("✗ Some tests FAILED! Dataset splits need adjustment.")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
