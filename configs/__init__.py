"""
Configuration module for unified experiment runner.
"""

import os

from .experiment_config import (
    ExperimentConfig,
    AblationExperimentConfig,
    VICComponents,
    RunMode
)

# Define data_dir and save_dir to ensure backward compatibility
# with code that imports configs and expects data_dir
# Note: This duplicates definitions from configs.py, but is necessary because
# Python imports the configs package (this __init__.py) when "import configs" is used,
# not the configs.py module.

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
