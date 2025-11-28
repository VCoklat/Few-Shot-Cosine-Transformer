import os

# Get base directory from environment variable or use default
# Can be set via: export FEW_SHOT_BASE_DIR=/path/to/Few-Shot-Cosine-Transformer
_base_dir = os.environ.get('FEW_SHOT_BASE_DIR', '/kaggle/working/Few-Shot-Cosine-Transformer')

save_dir                    = ''
data_dir = {}
data_dir['CUB']             = os.path.join(_base_dir, 'dataset/CUB/')
data_dir['miniImagenet']    = os.path.join(_base_dir, 'dataset/miniImagenet/')
data_dir['Omniglot']        = os.path.join(_base_dir, 'dataset/Omniglot/')
data_dir['emnist']          = os.path.join(_base_dir, 'dataset/emnist/')
data_dir['Yoga']            = os.path.join(_base_dir, 'dataset/Yoga/')
data_dir['CIFAR']           = os.path.join(_base_dir, 'dataset/CIFAR_FS/')
data_dir['HAM10000']        = os.path.join(_base_dir, 'dataset/HAM10000/')
data_dir['DatasetIndo']     = os.path.join(_base_dir, 'dataset/DatasetIndo/')