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

identity = lambda x:x


def _correct_image_path(original_path, data_file):
    """
    Correct image path if it doesn't exist by remapping to the current dataset location.
    
    This handles the case where JSON files contain absolute paths from a different machine.
    It extracts the relative path from 'dataset/' onwards and reconstructs using the 
    actual location of the data_file.
    
    Args:
        original_path: The original path from the JSON file
        data_file: The path to the JSON file being loaded
        
    Returns:
        Corrected path that should exist on the current system
    """
    if os.path.exists(original_path):
        return original_path
    
    # Try to find 'dataset/' in the path and extract the relative portion
    dataset_markers = ['/dataset/', '\\dataset\\']
    for marker in dataset_markers:
        if marker in original_path:
            # Get the relative path starting from the dataset name
            # e.g., "/old/path/dataset/DatasetIndo/train/..." -> "DatasetIndo/train/..."
            relative_path = original_path.split(marker, 1)[1]
            
            # Get the base directory from the data_file path
            # data_file is something like "/current/path/dataset/DatasetIndo/base.json"
            data_file_dir = os.path.dirname(os.path.abspath(data_file))
            
            # Find where 'dataset' folder is relative to data_file
            # Go up from data_file directory to find the dataset root
            current_dir = data_file_dir
            while current_dir and os.path.basename(current_dir) != 'dataset':
                parent = os.path.dirname(current_dir)
                if parent == current_dir:  # Reached root
                    break
                current_dir = parent
            
            if os.path.basename(current_dir) == 'dataset':
                new_path = os.path.join(current_dir, relative_path)
                if os.path.exists(new_path):
                    return new_path
    
    # Alternative: try to reconstruct path relative to data_file's parent directories
    # This handles cases where the structure is consistent but base paths differ
    path_parts = original_path.replace('\\', '/').split('/')
    data_file_abs = os.path.abspath(data_file)
    
    # Find common structure (e.g., dataset name, train/test/valid, class folder, filename)
    # Look for the dataset folder name in the path
    for i, part in enumerate(path_parts):
        if part in ['train', 'test', 'valid', 'val']:
            # Reconstruct from this point
            relative_from_split = '/'.join(path_parts[i:])
            # data_file is in the dataset folder, so go up one level from its directory
            dataset_dir = os.path.dirname(data_file_abs)
            new_path = os.path.join(dataset_dir, relative_from_split)
            if os.path.exists(new_path):
                return new_path
    
    # Return original path if no correction found (will fail later with a clearer error)
    return original_path


class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        # Store data_file path for path correction and batch_size
        self._data_file = data_file
        self.batch_size = batch_size
        
        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            # Correct the path if it doesn't exist
            corrected_path = _correct_image_path(x, data_file)
            self.sub_meta[y].append(corrected_path)

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
        # Get a batch from the sub_dataloader
        images, labels = next(iter(self.sub_dataloader[i]))
        
        # If we got fewer samples than batch_size, sample with replacement to reach batch_size
        if images.shape[0] < self.batch_size:
            # Number of additional samples needed
            n_additional = self.batch_size - images.shape[0]
            
            # Sample with replacement from the available samples using torch for thread safety
            additional_indices = torch.randint(0, images.shape[0], (n_additional,))
            additional_images = images[additional_indices]
            additional_labels = labels[additional_indices]
            
            # Concatenate to get exactly batch_size samples
            images = torch.cat([images, additional_images], dim=0)
            labels = torch.cat([labels, additional_labels], dim=0)
        
        return images, labels

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
