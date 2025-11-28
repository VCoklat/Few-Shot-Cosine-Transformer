import os

# Get base directory from environment variable or auto-detect from configs.py location
# Can be set via: export FEW_SHOT_BASE_DIR=/path/to/Few-Shot-Cosine-Transformer
_default_base_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.environ.get('FEW_SHOT_BASE_DIR', _default_base_dir)

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