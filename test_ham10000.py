#!/usr/bin/env python3
"""
Simple test to verify HAM10000 dataset integration.
This test checks that:
1. HAM10000 is registered in configs.py
2. JSON files exist and are properly formatted
3. Dataset can be loaded with SetDataset class
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configs
from data.dataset import SetDataset
import torchvision.transforms as transforms

def test_ham10000_config():
    """Test that HAM10000 is properly configured."""
    print("=" * 60)
    print("Testing HAM10000 Dataset Integration")
    print("=" * 60)
    
    # Test 1: Check if HAM10000 is in configs
    print("\n✓ Test 1: Checking configs.py...")
    assert 'HAM10000' in configs.data_dir, "HAM10000 not found in configs.data_dir"
    print(f"  ✅ HAM10000 path: {configs.data_dir['HAM10000']}")
    
    # Test 2: Check if directory exists
    print("\n✓ Test 2: Checking dataset directory...")
    dataset_path = configs.data_dir['HAM10000']
    assert os.path.exists(dataset_path), f"Dataset directory not found: {dataset_path}"
    print(f"  ✅ Directory exists: {dataset_path}")
    
    # Test 3: Check JSON files
    print("\n✓ Test 3: Checking JSON files...")
    for split in ['base', 'val', 'novel']:
        json_file = os.path.join(dataset_path, f'{split}.json')
        assert os.path.exists(json_file), f"JSON file not found: {json_file}"
        
        # Load and validate JSON structure
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        required_keys = ['label_names', 'image_names', 'image_labels']
        for key in required_keys:
            assert key in data, f"Missing key '{key}' in {split}.json"
        
        assert len(data['image_names']) == len(data['image_labels']), \
            f"Mismatch between image_names and image_labels in {split}.json"
        
        print(f"  ✅ {split}.json: {len(data['image_names'])} images, {len(data['label_names'])} classes")
    
    # Test 4: Test data loading with SetDataset
    print("\n✓ Test 4: Testing dataset loading...")
    try:
        base_file = os.path.join(dataset_path, 'base.json')
        transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor()
        ])
        
        dataset = SetDataset(base_file, batch_size=5, transform=transform)
        print(f"  ✅ SetDataset created successfully")
        print(f"  ✅ Number of classes: {len(dataset)}")
        print(f"  ✅ Class labels: {dataset.class_labels[:5]}...")  # Show first 5
        
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nHAM10000 dataset is properly integrated.")
    print("You can now use it with commands like:")
    print("  python train.py --dataset HAM10000 --method FSCT_cosine")
    print()
    
    return True

if __name__ == "__main__":
    try:
        test_ham10000_config()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
