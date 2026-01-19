
import os
import json
import configs

# Fix paths because configs.py uses a default that might not match local environment
current_dir = os.getcwd()
# Assume we are in the root of the project
# We need to remap the values in configs.data_dir
real_data_dir = {}

for key, val in configs.data_dir.items():
    # val is like '/kaggle/working/Few-Shot-Cosine-Transformer/dataset/CUB/'
    # We want 'dataset/CUB/'
    # We can split by 'dataset/' to get the relative suffix
    if 'dataset/' in val:
        suffix = val.split('dataset/')[1]
        real_data_dir[key] = os.path.join(current_dir, 'dataset', suffix)
    else:
        real_data_dir[key] = val

def get_stats(file_path):
    if not os.path.exists(file_path):
        return "File Not Found", 0, 0
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        n_images = len(data['image_labels'])
        n_classes = len(set(data['image_labels']))
        
        return "OK", n_images, n_classes
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def get_stats_csv_miniimagenet(file_path):
    if not os.path.exists(file_path):
        return "Missing", 0, 0
    
    try:
        # miniImagenet csvs usually have filename,label
        # We need to count lines and unique labels
        labels = []
        with open(file_path, 'r') as f:
            # check header
            first_line = f.readline()
            has_header = 'filename' in first_line or 'label' in first_line
            
            if not has_header:
                f.seek(0)
                
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    labels.append(parts[1])
                    
        n_images = len(labels)
        n_classes = len(set(labels))
        return "OK (CSV)", n_images, n_classes
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

print(f"{'Dataset':<15} | {'Split':<10} | {'Images':<8} | {'Classes':<8} | {'Status'}")
print("-" * 65)

datasets = [
    ('DatasetIndo', 'base.json', 'val.json', 'novel.json'),
    ('CUB', 'base.json', 'val.json', 'novel.json'),
    ('CIFAR', 'base.json', 'val.json', 'novel.json'),
    ('miniImagenet', 'base.json', 'val.json', 'novel.json'), 
    ('Omniglot', 'base.json', 'val.json', 'novel.json'),
    ('emnist', 'base.json', 'val.json', 'novel.json'),
    ('Yoga', 'base.json', 'val.json', 'novel.json'),
    ('HAM10000', 'base.json', 'val.json', 'novel.json')
]

for name, base, val, novel in datasets:
    # Map name to config key
    config_key = name
    
    if config_key not in real_data_dir:
        if name == 'emnist' and 'emnist' in real_data_dir:
            config_key = 'emnist'
        else:
            # print(f"Skipping {name} (Not in configs)")
            continue
        
    dir_path = real_data_dir[config_key]
    
    splits = [('Train/Base', base), ('Val', val), ('Test/Novel', novel)]
    
    for split_name, file_name in splits:
        full_path = os.path.join(dir_path, file_name)
        
        status, n_imgs, n_cls = get_stats(full_path)
        
        # Special handling for miniImagenet which might use CSVs if JSONs are missing
        if name == 'miniImagenet' and status == "File Not Found":
             # Try CSV mapping
             csv_name = file_name.replace('base', 'train').replace('novel', 'test').replace('json', 'csv')
             full_path_csv = os.path.join(dir_path, csv_name)
             status, n_imgs, n_cls = get_stats_csv_miniimagenet(full_path_csv)
             
        # If still missing, check for files in the dir to help debug but just print status
        
        print(f"{name:<15} | {split_name:<10} | {n_imgs:<8} | {n_cls:<8} | {status}")
    print("-" * 65)
