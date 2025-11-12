"""Test script for HAM10000 dataset implementation.

This script demonstrates the usage of HAM10000Dataset class and validates
that it works correctly with the Few-Shot-Cosine-Transformer framework.
"""

import os
import sys
import torch
from torchvision import transforms

# Add parent directory to path to import data module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ham10000_dataset import HAM10000Dataset


def test_ham10000_dataset():
    """Test the HAM10000Dataset class with sample data structure."""
    
    print("=" * 70)
    print("HAM10000 Dataset Test")
    print("=" * 70)
    
    # Define paths (adjust these based on your setup)
    csv_file = 'dataset/HAM10000/HAM10000_metadata.csv'
    img_dirs = [
        'dataset/HAM10000/HAM10000_images_part_1',
        'dataset/HAM10000/HAM10000_images_part_2'
    ]
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"\n⚠️  CSV file not found: {csv_file}")
        print("Please download the HAM10000 dataset from Kaggle and run setup.")
        print("See dataset/HAM10000/README.md for instructions.")
        return False
    
    print(f"\n✓ Found CSV file: {csv_file}")
    
    # Check image directories
    img_dirs_exist = [os.path.exists(d) for d in img_dirs]
    if not any(img_dirs_exist):
        print(f"\n⚠️  Image directories not found:")
        for d in img_dirs:
            print(f"    - {d}")
        print("Please extract the image files. See dataset/HAM10000/README.md for instructions.")
        return False
    
    for d in img_dirs:
        if os.path.exists(d):
            print(f"✓ Found image directory: {d}")
    
    # Define transforms
    print("\n" + "-" * 70)
    print("Setting up transforms...")
    print("-" * 70)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("\nCreating HAM10000Dataset instance...")
    try:
        dataset = HAM10000Dataset(
            csv_file=csv_file,
            img_dirs=img_dirs,
            transform=transform,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"✓ Dataset created successfully!")
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        return False
    
    # Print dataset information
    print("\n" + "-" * 70)
    print("Dataset Information")
    print("-" * 70)
    print(f"Total samples: {len(dataset)}")
    
    # Get label mapping
    label_mapping = dataset.get_label_mapping()
    print(f"\nDiagnostic categories ({len(label_mapping)}):")
    for idx, name in label_mapping.items():
        print(f"  {idx}: {name}")
    
    # Get class distribution
    print("\nClass distribution:")
    class_counts = dataset.get_class_counts()
    for cls, count in class_counts.items():
        percentage = (count / len(dataset)) * 100
        print(f"  {cls}: {count} images ({percentage:.1f}%)")
    
    # Test loading a sample
    print("\n" + "-" * 70)
    print("Testing sample loading...")
    print("-" * 70)
    
    try:
        image, label = dataset[0]
        print(f"✓ Successfully loaded sample 0")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Label: {label} ({label_mapping[label]})")
        
        # Test a few more samples
        print("\nTesting additional samples...")
        for i in range(min(5, len(dataset))):
            img, lbl = dataset[i]
            print(f"  Sample {i}: image {img.shape}, label {lbl} ({label_mapping[lbl]})")
        
        print(f"\n✓ All sample tests passed!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error loading sample: {e}")
        print("Some images may be missing. Please verify your dataset setup.")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False
    
    # Test with different indices
    print("\n" + "-" * 70)
    print("Testing random access...")
    print("-" * 70)
    
    try:
        import random
        random.seed(42)
        test_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
        
        for idx in test_indices:
            img, lbl = dataset[idx]
            # Just verify it doesn't crash
        
        print(f"✓ Random access test passed (tested {len(test_indices)} samples)")
        
    except Exception as e:
        print(f"✗ Random access test failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All tests passed successfully!")
    print("=" * 70)
    print("\nThe HAM10000Dataset implementation is working correctly.")
    print("You can now use it for few-shot learning experiments.")
    
    return True


def print_usage_example():
    """Print example usage code."""
    print("\n" + "=" * 70)
    print("Usage Example")
    print("=" * 70)
    
    example_code = '''
from data.ham10000_dataset import HAM10000Dataset
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = HAM10000Dataset(
    csv_file='dataset/HAM10000/HAM10000_metadata.csv',
    img_dirs=[
        'dataset/HAM10000/HAM10000_images_part_1',
        'dataset/HAM10000/HAM10000_images_part_2'
    ],
    transform=transform,
    device='cuda'
)

# Use in DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate
for images, labels in loader:
    # Your training/testing code here
    pass
'''
    
    print(example_code)
    
    print("\nFor few-shot learning with this framework:")
    print("  python train.py --dataset HAM10000 --method FSCT_cosine --backbone ResNet18")


if __name__ == '__main__':
    print("\nHAM10000 Dataset Implementation Test")
    print("This script tests the HAM10000Dataset class implementation.\n")
    
    success = test_ham10000_dataset()
    
    if success:
        print_usage_example()
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("Refer to dataset/HAM10000/README.md for setup instructions.")
        sys.exit(1)
