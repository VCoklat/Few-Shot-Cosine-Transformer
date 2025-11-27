# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import pdb
import torch
from PIL import Image
import json
import numpy as np
import random
import torchvision.transforms as transforms
import os
import cv2 as cv
from pathlib import Path

identity = lambda x:x

# Maximum depth to search upward for dataset files
MAX_SEARCH_DEPTH = 5

# Common dataset directory patterns to look for in paths
DATASET_DIR_PATTERNS = ['dataset', 'Dataset', 'data', 'Data']


def resolve_image_path(image_path, data_file_dir):
    """
    Resolve image path that may be absolute or relative.
    If the absolute path doesn't exist, try to make it relative to the JSON file location.
    """
    if os.path.exists(image_path):
        return image_path
    
    # Normalize path and split into components using pathlib for robustness
    normalized_path = Path(image_path)
    path_parts = normalized_path.parts
    
    # Try to extract relative path from absolute path
    # Look for common dataset directory patterns as complete directory components
    for i, part in enumerate(path_parts):
        if part in DATASET_DIR_PATTERNS:
            # Get the relative path starting from dataset directory
            rel_path = str(Path(*path_parts[i:]))
            # Go up from data_file_dir to find dataset directory
            base_dir = data_file_dir
            depth = 0
            while base_dir and not os.path.exists(os.path.join(base_dir, rel_path)):
                parent = os.path.dirname(base_dir)
                if parent == base_dir:  # Reached root
                    break
                base_dir = parent
                depth += 1
                if depth >= MAX_SEARCH_DEPTH:
                    break
            
            resolved = os.path.join(base_dir, rel_path)
            if os.path.exists(resolved):
                return resolved
    
    # If still not found, try progressively shorter paths
    for i in range(len(path_parts)):
        remaining_parts = path_parts[i:]
        if not remaining_parts:
            continue
        rel_path = str(Path(*remaining_parts))
        # Search upward from data_file_dir
        search_dir = data_file_dir
        for _ in range(MAX_SEARCH_DEPTH):
            candidate = os.path.join(search_dir, rel_path)
            if os.path.exists(candidate):
                return candidate
            parent = os.path.dirname(search_dir)
            if parent == search_dir:
                break
            search_dir = parent
    
    # Return original path if nothing works (will raise error on access)
    return image_path


class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        
        # Store the directory containing the JSON file for path resolution
        self.data_file_dir = os.path.dirname(os.path.abspath(data_file))
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            # Resolve the image path relative to JSON file location if needed
            resolved_path = resolve_image_path(x, self.data_file_dir)
            self.sub_meta[y].append(resolved_path)

        self.sub_dataloader = [] 
        
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(
                self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
        # pdb.set_trace()

    def __getitem__(self,i):
        data, label = next(iter(self.sub_dataloader[i]))
        # Create new tensors with independent, resizable storage to fix
        # "Trying to resize storage that is not resizable" error when using
        # DataLoader with num_workers > 0. The clone() creates a copy,
        # but we need to ensure the storage is independent, so we also
        # call contiguous() which may create a new storage if needed.
        if torch.is_tensor(data):
            data = data.clone().contiguous()
        if torch.is_tensor(label):
            label = label.clone().contiguous()
        return data, label

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self,i):
        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)
        
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)
        
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
