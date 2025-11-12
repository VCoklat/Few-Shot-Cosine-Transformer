"""Generate JSON metadata files for HAM10000 dataset.

This script creates base.json, val.json, and novel.json files for the HAM10000 dataset
following the Few-Shot-Cosine-Transformer repository's data format.

The HAM10000 dataset contains 7 diagnostic categories:
    - akiec: Actinic keratoses and intraepithelial carcinoma
    - bcc: Basal cell carcinoma
    - bkl: Benign keratosis-like lesions
    - df: Dermatofibroma
    - mel: Melanoma
    - nv: Melanocytic nevi
    - vasc: Vascular lesions

Usage:
    1. Download HAM10000 dataset from Kaggle
    2. Extract images to HAM10000_images_part_1/ and HAM10000_images_part_2/
    3. Place HAM10000_metadata.csv in the same directory as this script
    4. Run: python write_ham10000_filelist.py
"""

import os
import json
import random
import pandas as pd
import numpy as np
from os.path import join, exists


def write_ham10000_filelist():
    """Generate JSON metadata files for HAM10000 dataset split into base/val/novel sets."""
    
    cwd = os.getcwd()
    csv_file = join(cwd, 'HAM10000_metadata.csv')
    
    # Check if CSV file exists
    if not exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please download HAM10000_metadata.csv from Kaggle and place it in this directory.")
        return
    
    # Image directories (HAM10000 dataset is split into two parts)
    img_dirs = [
        join(cwd, 'HAM10000_images_part_1'),
        join(cwd, 'HAM10000_images_part_2')
    ]
    
    # Check if image directories exist
    for img_dir in img_dirs:
        if not exists(img_dir):
            print(f"Warning: {img_dir} not found!")
    
    savedir = './'
    dataset_list = ['base', 'val', 'novel']
    
    # Load the CSV file
    print("Loading HAM10000 metadata...")
    df = pd.read_csv(csv_file)
    
    # Get unique classes
    classes = sorted(df['dx'].unique())
    print(f"Found {len(classes)} classes: {classes}")
    
    # Create a mapping of class names to indices
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Group images by class
    class_images = {cls: [] for cls in classes}
    
    for _, row in df.iterrows():
        img_id = row['image_id']
        dx = row['dx']
        
        # Find the image file
        img_found = False
        for img_dir in img_dirs:
            img_path = join(img_dir, f"{img_id}.jpg")
            if exists(img_path):
                class_images[dx].append(img_path)
                img_found = True
                break
        
        if not img_found:
            # Try without .jpg extension in case it's already there
            for img_dir in img_dirs:
                img_path = join(img_dir, img_id)
                if exists(img_path):
                    class_images[dx].append(img_path)
                    break
    
    # Shuffle images within each class
    for cls in classes:
        random.shuffle(class_images[cls])
        print(f"Class '{cls}': {len(class_images[cls])} images")
    
    # Split classes into base/val/novel sets
    # Using a split similar to miniImagenet: base=5 classes, val=1 class, novel=1 class
    # For 7 classes: base=4, val=2, novel=1
    num_classes = len(classes)
    
    if num_classes >= 7:
        # Standard split for 7 classes
        base_classes = classes[:4]      # 4 classes for training
        val_classes = classes[4:6]      # 2 classes for validation
        novel_classes = classes[6:]     # 1 class for testing
    else:
        # Fallback for fewer classes
        base_size = max(1, num_classes // 2)
        val_size = max(1, (num_classes - base_size) // 2)
        base_classes = classes[:base_size]
        val_classes = classes[base_size:base_size + val_size]
        novel_classes = classes[base_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Base classes ({len(base_classes)}): {base_classes}")
    print(f"  Val classes ({len(val_classes)}): {val_classes}")
    print(f"  Novel classes ({len(novel_classes)}): {novel_classes}")
    
    # Create JSON files for each split
    splits = {
        'base': base_classes,
        'val': val_classes,
        'novel': novel_classes
    }
    
    for dataset, split_classes in splits.items():
        file_list = []
        label_list = []
        
        for cls in split_classes:
            cls_idx = class_to_idx[cls]
            images = class_images[cls]
            file_list.extend(images)
            label_list.extend([cls_idx] * len(images))
        
        # Create JSON file
        json_data = {
            'label_names': classes,
            'image_names': file_list,
            'image_labels': label_list
        }
        
        output_file = join(savedir, f'{dataset}.json')
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n{dataset}.json created:")
        print(f"  Classes: {len(split_classes)}")
        print(f"  Total images: {len(file_list)}")
        print(f"  Saved to: {output_file}")
    
    print("\nDone! JSON files created successfully.")
    print("\nNext steps:")
    print("1. Add 'HAM10000' to configs.py data_dir dictionary")
    print("2. Use --dataset HAM10000 when running train.py or test.py")


if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    write_ham10000_filelist()
