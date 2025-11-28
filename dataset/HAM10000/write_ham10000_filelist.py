#!/usr/bin/env python3
"""
Script to process HAM10000 skin cancer dataset for few-shot learning.
This script reads the metadata CSV and generates base.json, val.json, and novel.json
files compatible with the Few-Shot Cosine Transformer framework.

Dataset structure expected:
- CSV metadata file with image_id and dx (diagnosis) columns
- Images organized in Dataset folder by class (akiec, bcc, bkl, df, mel, nv, vasc)
"""

import numpy as np
import pandas as pd
import os
import json
import random
from os.path import join, isfile, isdir

# --- Configuration ---
# Path to the combined images folder (organized by class)
dataset_base_path = 'dataset/HAM10000/Dataset'

# Path to the metadata CSV file
image_list_path = 'HAM10000_metadata.csv'

cwd = os.getcwd()
savedir = './'
dataset_list = ['base', 'val', 'novel']

def load_ham10000_data(csv_path, img_dir):
    """
    Load HAM10000 dataset from CSV file and image directory.
    
    Args:
        csv_path: Path to metadata CSV file
        img_dir: Path to image directory (organized by class folders)
    
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
    
    # Build full image paths - images are organized by class folder
    image_paths = []
    not_found_count = 0
    for idx, row in df.iterrows():
        img_id = row['image_id']
        label = row[label_col]
        
        # Remove .jpg extension if present
        img_id_clean = img_id.replace('.jpg', '')
        
        # Image path: Dataset/<class>/<image_id>.jpg
        img_path = os.path.join(img_dir, label, f"{img_id_clean}.jpg")
        
        if os.path.exists(img_path):
            image_paths.append(img_path)
        else:
            # If running without actual images, use the path anyway
            image_paths.append(img_path)
            not_found_count += 1
    
    if not_found_count > 0:
        print(f"⚠️  Warning: {not_found_count} images not found")
    
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
    
    try:
        # Load the dataset
        print("\n📂 Loading HAM10000 dataset...")
        df = load_ham10000_data(image_list_path, dataset_base_path)
        
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
        print("  1. The metadata CSV file path is correct")
        print("  2. The Dataset directory exists with class folders")
        print("  3. You have downloaded the HAM10000 dataset")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
