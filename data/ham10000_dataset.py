"""HAM10000 Dataset implementation for Few-Shot Learning.

This module provides a PyTorch Dataset class for the HAM10000 skin lesion dataset,
which contains dermatoscopic images of various skin lesions across 7 diagnostic categories.
"""

import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):
    """A PyTorch Dataset class for the HAM10000 skin lesion dataset.
    
    This class handles loading and preprocessing of the HAM10000 dataset, which contains
    dermatoscopic images of various skin lesions across 7 diagnostic categories.
    
    Args:
        csv_file (str): Path to the CSV file containing image metadata and labels.
        img_dirs (list): List of directory paths containing the image files.
        transform (callable, optional): Optional transform to be applied on a sample.
            Defaults to None.
        device (str, optional): Device to store the tensors on ('cuda' or 'cpu').
            Defaults to 'cuda'.
    
    Attributes:
        data (pandas.DataFrame): The loaded CSV data containing image metadata.
        img_dirs (list): List of directories containing image files.
        transform (callable): Transform to be applied to images.
        device (str): Device for tensor storage.
        label_encoder (LabelEncoder): Scikit-learn label encoder for categorical labels.
    
    Returns:
        tuple: A tuple containing:
            - image (Tensor): The processed image
            - label (int): The encoded label
    
    Raises:
        FileNotFoundError: If an image file cannot be found in any of the provided directories.
    """
    
    def __init__(self, csv_file, img_dirs, transform=None, device='cuda'):
        """Initialize the HAM10000 dataset.
        
        Args:
            csv_file (str): Path to the CSV file containing image metadata and labels.
            img_dirs (list): List of directory paths containing the image files.
            transform (callable, optional): Optional transform to be applied on a sample.
            device (str, optional): Device to store the tensors on ('cuda' or 'cpu').
        """
        self.data = pd.read_csv(csv_file)
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.transform = transform
        self.device = device
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.data['encoded_label'] = self.label_encoder.fit_transform(self.data['dx'])
    
    def __len__(self):
        """Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing:
                - image (Tensor): The processed image
                - label (int): The encoded label
        
        Raises:
            FileNotFoundError: If the image file cannot be found in any directory.
        """
        img_name = self.data.iloc[idx]['image_id'] + '.jpg'
        
        # Search for the image in all provided directories
        for img_dir in self.img_dirs:
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                label = self.data.iloc[idx]['encoded_label']
                return image, label
        
        # If image not found in any directory, raise an error
        raise FileNotFoundError(f"Image {img_name} not found in directories {self.img_dirs}")
    
    def get_label_mapping(self):
        """Get the mapping between encoded labels and original class names.
        
        Returns:
            dict: Dictionary mapping encoded labels (int) to original class names (str).
        """
        return dict(enumerate(self.label_encoder.classes_))
    
    def get_class_counts(self):
        """Get the count of samples for each class.
        
        Returns:
            pandas.Series: Series containing counts for each diagnostic category.
        """
        return self.data['dx'].value_counts()
