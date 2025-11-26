#!/usr/bin/env python3
"""
Script to process HAM10000 skin cancer dataset for few-shot learning.
This script reads the image list CSV and generates base.json, val.json, and novel.json
files compatible with the Few-Shot Cosine Transformer framework.

Dataset structure expected:
- CSV file with image names and labels
- Images in HAM10000_images_part_1 and HAM10000_images_part_2 directories
"""

import numpy as np
import pandas as pd
import os
import json
import random
from os.path import join, isfile, isdir

# --- Configuration ---
# These paths should be configured by the user
# Default paths assume Kaggle dataset structure
dataset_base_path = '/kaggle/input/skin-cancer-the-ham10000-dataset/'
path_part1 = os.path.join(dataset_base_path, 'HAM10000_images_part_1')
path_part2 = os.path.join(dataset_base_path, 'HAM10000_images_part_2')

# Path to the CSV file listing 1000 images
# This should be created by the user before running this script
image_list_path = '/kaggle/input/d/rafiarrantisi/my-ham1000-final-list/final_1000_image_list.csv'

# Alternative: Use local paths if running locally
# Uncomment and modify these paths as needed:
# dataset_base_path = './HAM10000_dataset/'
# path_part1 = os.path.join(dataset_base_path, 'HAM10000_images_part_1')
# path_part2 = os.path.join(dataset_base_path, 'HAM10000_images_part_2')
# image_list_path = './final_1000_image_list.csv'

cwd = os.getcwd()
savedir = './'
dataset_list = ['base', 'val', 'novel']

def load_ham10000_data(csv_path, img_dir_part1, img_dir_part2):
    """
    Load HAM10000 dataset from CSV file and image directories.
    
    Args:
        csv_path: Path to CSV file with image list
        img_dir_part1: Path to first image directory
        img_dir_part2: Path to second image directory
    
    Returns:
        DataFrame with image paths and labels
    """
    # Load the CSV file
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} images from CSV file")
    
    # Ensure required columns exist
    if 'image_id' not in df.columns:
        # Try to find image_id or image column
        if 'image' in df.columns:
            df['image_id'] = df['image']
        elif 'image_name' in df.columns:
            df['image_id'] = df['image_name']
        else:
            raise ValueError("CSV must contain 'image_id', 'image', or 'image_name' column")
    
    if 'dx' not in df.columns and 'label' not in df.columns:
        raise ValueError("CSV must contain 'dx' or 'label' column for class labels")
    
    # Use 'dx' as label if it exists, otherwise use 'label'
    label_col = 'dx' if 'dx' in df.columns else 'label'
    
    # Build full image paths
    image_paths = []
    for img_id in df['image_id']:
        # Remove .jpg extension if present
        img_id_clean = img_id.replace('.jpg', '')
        
        # Try to find image in both directories
        path1 = os.path.join(img_dir_part1, f"{img_id_clean}.jpg")
        path2 = os.path.join(img_dir_part2, f"{img_id_clean}.jpg")
        
        if os.path.exists(path1):
            image_paths.append(path1)
        elif os.path.exists(path2):
            image_paths.append(path2)
        else:
            # If running without actual images, use the path anyway
            # This allows the script to generate JSON files for structure
            image_paths.append(path1)
            print(f"⚠️  Warning: Image not found: {img_id_clean}.jpg")
    
    df['image_path'] = image_paths
    df['label'] = df[label_col]
    
    return df

def split_dataset(df, base_ratio=0.64, val_ratio=0.16, novel_ratio=0.20):
    """
    Split dataset into base, val, and novel sets based on classes.
    
    Args:
        df: DataFrame with image data
        base_ratio: Proportion of classes for base set (training)
        val_ratio: Proportion of classes for validation set
        novel_ratio: Proportion of classes for novel set (testing)
    
    Returns:
        Dictionary with 'base', 'val', and 'novel' splits
    """
    # Get unique classes and shuffle them
    unique_classes = df['label'].unique()
    random.shuffle(unique_classes)
    
    n_classes = len(unique_classes)
    n_base = int(n_classes * base_ratio)
    n_val = int(n_classes * val_ratio)
    
    # Split classes
    base_classes = unique_classes[:n_base]
    val_classes = unique_classes[n_base:n_base + n_val]
    novel_classes = unique_classes[n_base + n_val:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Total classes: {n_classes}")
    print(f"   Base classes: {len(base_classes)} ({base_ratio*100:.0f}%)")
    print(f"   Val classes: {len(val_classes)} ({val_ratio*100:.0f}%)")
    print(f"   Novel classes: {len(novel_classes)} ({novel_ratio*100:.0f}%)")
    
    # Create splits
    splits = {
        'base': df[df['label'].isin(base_classes)].copy(),
        'val': df[df['label'].isin(val_classes)].copy(),
        'novel': df[df['label'].isin(novel_classes)].copy()
    }
    
    # Shuffle images within each split
    for split_name in splits:
        splits[split_name] = splits[split_name].sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"   {split_name.capitalize()} images: {len(splits[split_name])}")
    
    return splits

def create_json_files(splits, all_classes, savedir):
    """
    Create JSON files for base, val, and novel splits.
    
    Args:
        splits: Dictionary with 'base', 'val', 'novel' DataFrames
        all_classes: List of all class names
        savedir: Directory to save JSON files
    """
    # Create label encoding
    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_classes))}
    
    for split_name, df_split in splits.items():
        if len(df_split) == 0:
            print(f"⚠️  Warning: {split_name} split is empty, skipping...")
            continue
        
        # Get image paths and labels
        image_paths = df_split['image_path'].tolist()
        labels = [label_to_idx[label] for label in df_split['label']]
        
        # Create JSON data
        json_data = {
            "label_names": sorted(all_classes),
            "image_names": image_paths,
            "image_labels": labels
        }
        
        # Save JSON file
        output_path = os.path.join(savedir, f"{split_name}.json")
        with open(output_path, 'w') as f:
            json.dump(json_data, f)
        
        print(f"✅ {split_name}.json created with {len(image_paths)} images")

def main():
    """Main function to process HAM10000 dataset."""
    print("=" * 60)
    print("HAM10000 Dataset Processing for Few-Shot Learning")
    print("=" * 60)
    
    # Check if running in Kaggle environment
    is_kaggle = os.path.exists('/kaggle')
    
    if not is_kaggle:
        print("\n⚠️  Note: Not running in Kaggle environment.")
        print("Please configure the paths at the top of this script:")
        print("  - dataset_base_path")
        print("  - image_list_path")
        print("\nOr run this script in Kaggle with the HAM10000 dataset.")
        
        # For demonstration, create sample JSON files with example structure
        print("\n📝 Creating sample JSON files for demonstration...")
        
        # Create sample data
        sample_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        sample_data = {
            'base': {
                'image_names': [f'./HAM10000_images_part_1/sample_{i}.jpg' for i in range(10)],
                'image_labels': [i % 5 for i in range(10)]
            },
            'val': {
                'image_names': [f'./HAM10000_images_part_1/sample_{i}.jpg' for i in range(10, 13)],
                'image_labels': [5, 5, 5]
            },
            'novel': {
                'image_names': [f'./HAM10000_images_part_1/sample_{i}.jpg' for i in range(13, 16)],
                'image_labels': [6, 6, 6]
            }
        }
        
        for split_name in dataset_list:
            json_data = {
                "label_names": sample_classes,
                "image_names": sample_data[split_name]['image_names'],
                "image_labels": sample_data[split_name]['image_labels']
            }
            
            output_path = os.path.join(savedir, f"{split_name}.json")
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"✅ Sample {split_name}.json created")
        
        print("\n✅ Sample JSON files created successfully!")
        print("Replace these with actual data when you have the dataset.")
        return
    
    try:
        # Load the dataset
        print("\n📂 Loading HAM10000 dataset...")
        df = load_ham10000_data(image_list_path, path_part1, path_part2)
        
        # Get all unique classes
        all_classes = sorted(df['label'].unique().tolist())
        print(f"\n🏷️  Found {len(all_classes)} classes: {all_classes}")
        
        # Split dataset into base, val, novel
        print("\n✂️  Splitting dataset...")
        splits = split_dataset(df)
        
        # Create JSON files
        print("\n💾 Creating JSON files...")
        create_json_files(splits, all_classes, savedir)
        
        print("\n" + "=" * 60)
        print("✅ HAM10000 dataset processing completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        for split in dataset_list:
            print(f"  - {split}.json")
        print("\nYou can now use the HAM10000 dataset with the Few-Shot Cosine Transformer.")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. The CSV file path is correct")
        print("  2. The image directories exist")
        print("  3. You have downloaded the HAM10000 dataset")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
