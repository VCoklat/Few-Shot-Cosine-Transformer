#!/usr/bin/env python3
"""
Script to organize HAM10000 dataset images into class-based subdirectories.

This script reads the HAM10000_metadata file and moves/copies images from the 
flat Dataset directory into subdirectories named by diagnosis (dx) class.

Usage:
    python organize_HAM10000.py
    
The script will create the following structure:
    Dataset/
        akiec/
            ISIC_XXXXXXX.jpg
            ...
        bcc/
            ...
        bkl/
            ...
        df/
            ...
        mel/
            ...
        nv/
            ...
        vasc/
            ...
"""

import os
import shutil
import csv
from collections import defaultdict


def organize_ham10000_dataset(metadata_path, dataset_dir, move_files=True):
    """
    Organize HAM10000 images into class-based subdirectories.
    
    Args:
        metadata_path: Path to HAM10000_metadata CSV file
        dataset_dir: Path to the Dataset directory containing images
        move_files: If True, move files; if False, copy files
    """
    # Read metadata file
    image_to_class = {}
    class_counts = defaultdict(int)
    
    with open(metadata_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        # Find column indices
        image_id_idx = header.index('image_id')
        dx_idx = header.index('dx')
        
        for row in reader:
            if len(row) > max(image_id_idx, dx_idx):
                image_id = row[image_id_idx]
                dx = row[dx_idx]
                image_to_class[image_id] = dx
                class_counts[dx] += 1
    
    print(f"Read {len(image_to_class)} entries from metadata file")
    print("\nClass distribution in metadata:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count} images")
    
    # Create class subdirectories
    classes = set(image_to_class.values())
    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory: {class_dir}")
    
    # Process images
    moved_count = 0
    not_found_count = 0
    already_organized_count = 0
    
    for image_id, cls in image_to_class.items():
        # Try different file extensions
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            src_path = os.path.join(dataset_dir, image_id + ext)
            if os.path.exists(src_path):
                # Destination path
                dst_dir = os.path.join(dataset_dir, cls)
                dst_path = os.path.join(dst_dir, image_id + ext)
                
                # Skip if already in correct location
                if os.path.exists(dst_path):
                    already_organized_count += 1
                    break
                
                # Move or copy file
                if move_files:
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                moved_count += 1
                break
        else:
            # Check if file already exists in a class subdirectory
            found_in_subdir = False
            for existing_cls in classes:
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    check_path = os.path.join(dataset_dir, existing_cls, image_id + ext)
                    if os.path.exists(check_path):
                        found_in_subdir = True
                        already_organized_count += 1
                        break
                if found_in_subdir:
                    break
            
            if not found_in_subdir:
                not_found_count += 1
                if not_found_count <= 5:  # Only show first 5 warnings
                    print(f"Warning: Image not found: {image_id}")
    
    print(f"\n{'Moved' if move_files else 'Copied'}: {moved_count} images")
    print(f"Already organized: {already_organized_count} images")
    if not_found_count > 0:
        print(f"Not found: {not_found_count} images")
    
    # Verify final structure
    print("\nFinal directory structure:")
    total_in_subdirs = 0
    for cls in sorted(classes):
        class_dir = os.path.join(dataset_dir, cls)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if os.path.isfile(os.path.join(class_dir, f))])
            total_in_subdirs += count
            print(f"  {cls}/: {count} images")
    
    print(f"\nTotal images in class subdirectories: {total_in_subdirs}")
    
    return moved_count, not_found_count


def main():
    """Main function to organize HAM10000 dataset."""
    print("=" * 60)
    print("HAM10000 Dataset Organization Script")
    print("=" * 60)
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    metadata_path = os.path.join(script_dir, 'HAM10000_metadata')
    dataset_dir = os.path.join(script_dir, 'Dataset')
    
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        return 1
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    print(f"\nMetadata file: {metadata_path}")
    print(f"Dataset directory: {dataset_dir}")
    
    # Organize the dataset
    print("\nOrganizing images into class subdirectories...")
    moved_count, not_found_count = organize_ham10000_dataset(
        metadata_path, dataset_dir, move_files=True
    )
    
    if moved_count > 0 or not_found_count == 0:
        print("\n" + "=" * 60)
        print("Dataset organization completed successfully!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("Warning: No images were moved. Check your dataset directory.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
