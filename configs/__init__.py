"""
Configuration module for unified experiment runner.
"""

import os
import sys

from .experiment_config import (
    ExperimentConfig,
    AblationExperimentConfig,
    VICComponents,
    RunMode
)

# Import data_dir and save_dir from the top-level configs.py module
# This ensures backward compatibility with code that imports configs and expects data_dir
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Get base directory from environment variable or use default
_base_dir = os.environ.get('FEW_SHOT_BASE_DIR', '/kaggle/working/Few-Shot-Cosine-Transformer')

save_dir = ''
data_dir = {}
data_dir['CUB'] = os.path.join(_base_dir, 'dataset/CUB/')
data_dir['miniImagenet'] = os.path.join(_base_dir, 'dataset/miniImagenet/')
data_dir['Omniglot'] = os.path.join(_base_dir, 'dataset/Omniglot/')
data_dir['emnist'] = os.path.join(_base_dir, 'dataset/emnist/')
data_dir['Yoga'] = os.path.join(_base_dir, 'dataset/Yoga/')
data_dir['CIFAR'] = os.path.join(_base_dir, 'dataset/CIFAR_FS/')
data_dir['HAM10000'] = os.path.join(_base_dir, 'dataset/HAM10000/')
data_dir['DatasetIndo'] = os.path.join(_base_dir, 'dataset/DatasetIndo/')

__all__ = [
    'ExperimentConfig',
    'AblationExperimentConfig',
    'VICComponents',
    'RunMode',
    'data_dir',
    'save_dir'
]
