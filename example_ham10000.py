"""Example usage of HAM10000 dataset with Few-Shot-Cosine-Transformer.

This script demonstrates how to use the HAM10000 skin lesion dataset
for few-shot learning experiments using the FSCT framework.
"""

import os
import sys
import torch
from torchvision import transforms

# Ensure the parent directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ham10000_dataset import HAM10000Dataset


def create_ham10000_example():
    """Create and demonstrate HAM10000 dataset usage."""
    
    print("="*80)
    print("HAM10000 Dataset - Example Usage")
    print("="*80)
    
    # Configuration
    csv_file = 'dataset/HAM10000/HAM10000_metadata.csv'
    img_dirs = [
        'dataset/HAM10000/HAM10000_images_part_1',
        'dataset/HAM10000/HAM10000_images_part_2'
    ]
    
    # Check if dataset is available
    if not os.path.exists(csv_file):
        print("\n⚠️  HAM10000 dataset not found!")
        print("\nTo use this example:")
        print("1. Download HAM10000 from Kaggle:")
        print("   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("2. Extract images to dataset/HAM10000/")
        print("3. Place HAM10000_metadata.csv in dataset/HAM10000/")
        print("4. Run: cd dataset/HAM10000 && python write_ham10000_filelist.py")
        print("\nSee dataset/HAM10000/README.md for detailed instructions.")
        return
    
    # Define image transformations
    print("\n1. Setting up image transformations...")
    print("-"*80)
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("✓ Transforms configured:")
    print("  - Train: Resize(224) + RandomFlip + ColorJitter + Normalize")
    print("  - Test:  Resize(224) + Normalize")
    
    # Create dataset instance
    print("\n2. Creating HAM10000 dataset...")
    print("-"*80)
    
    try:
        dataset = HAM10000Dataset(
            csv_file=csv_file,
            img_dirs=img_dirs,
            transform=train_transform,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Total images: {len(dataset)}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Display dataset statistics
    print("\n3. Dataset Statistics")
    print("-"*80)
    
    label_mapping = dataset.get_label_mapping()
    class_counts = dataset.get_class_counts()
    
    print(f"\nDiagnostic Categories ({len(label_mapping)}):")
    category_descriptions = {
        'akiec': 'Actinic keratoses and intraepithelial carcinoma',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis-like lesions',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
    }
    
    for idx, name in sorted(label_mapping.items()):
        count = class_counts.get(name, 0)
        percentage = (count / len(dataset)) * 100
        desc = category_descriptions.get(name, 'Unknown')
        print(f"  [{idx}] {name:6s} - {desc:50s} ({count:5d} images, {percentage:5.1f}%)")
    
    # Demonstrate data loading
    print("\n4. Loading Sample Images")
    print("-"*80)
    
    print("\nLoading first 5 samples:")
    for i in range(min(5, len(dataset))):
        try:
            image, label = dataset[i]
            class_name = label_mapping[label]
            print(f"  Sample {i}: shape={image.shape}, label={label} ({class_name})")
        except Exception as e:
            print(f"  Sample {i}: Error - {e}")
    
    # Show usage with PyTorch DataLoader
    print("\n5. Using with PyTorch DataLoader")
    print("-"*80)
    
    from torch.utils.data import DataLoader, SubsetRandomSampler
    import numpy as np
    
    # Create a small subset for demonstration
    indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
    sampler = SubsetRandomSampler(indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Created DataLoader:")
    print(f"  Batch size: 16")
    print(f"  Subset size: {len(indices)}")
    print(f"  Number of batches: {len(dataloader)}")
    
    # Load one batch
    try:
        images, labels = next(iter(dataloader))
        print(f"\nLoaded one batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels in batch: {labels.tolist()}")
    except Exception as e:
        print(f"  Error loading batch: {e}")
    
    # Show few-shot learning setup
    print("\n6. Few-Shot Learning Setup")
    print("-"*80)
    
    print("\nRecommended configurations for HAM10000:")
    print("\nBasic few-shot (testing):")
    print("  python train.py --dataset HAM10000 --method FSCT_cosine \\")
    print("                  --backbone ResNet18 --n_way 5 --k_shot 5 \\")
    print("                  --train_aug 1 --num_epoch 50")
    
    print("\nAdvanced with ProFONet (better performance):")
    print("  python train.py --dataset HAM10000 --method FSCT_ProFONet \\")
    print("                  --backbone ResNet18 --n_way 5 --k_shot 5 \\")
    print("                  --train_aug 1 --num_epoch 50")
    
    print("\n1-shot learning (challenging):")
    print("  python train.py --dataset HAM10000 --method FSCT_cosine \\")
    print("                  --backbone ResNet18 --n_way 5 --k_shot 1 \\")
    print("                  --train_aug 1 --num_epoch 50")
    
    print("\nTesting:")
    print("  python test.py --dataset HAM10000 --method FSCT_cosine \\")
    print("                 --backbone ResNet18 --n_way 5 --k_shot 5")
    
    # Important notes
    print("\n7. Important Notes")
    print("-"*80)
    
    print("\n⚠️  Dataset Characteristics:")
    print("  - Highly imbalanced (nv: 67%, df: 1%)")
    print("  - Medical domain - requires careful evaluation")
    print("  - 7 classes split into: 4 base, 2 val, 1 novel")
    print("  - Images from multiple sources (different quality)")
    
    print("\n💡 Tips for Best Results:")
    print("  - Use data augmentation (--train_aug 1)")
    print("  - Consider ResNet18 or ResNet34 as backbone")
    print("  - Start with 5-way 5-shot for baseline")
    print("  - Monitor validation accuracy on val set")
    print("  - Use ProFONet method for better stability")
    
    print("\n📚 References:")
    print("  - Dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    print("  - Paper: Tschandl et al., Scientific Data 2018")
    print("  - See dataset/HAM10000/README.md for more details")
    
    print("\n" + "="*80)
    print("✓ Example completed successfully!")
    print("="*80)


if __name__ == '__main__':
    create_ham10000_example()
