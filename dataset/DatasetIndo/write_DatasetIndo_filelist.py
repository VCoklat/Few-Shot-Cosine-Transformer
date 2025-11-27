import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json

cwd = os.getcwd()

# Define mappings between splits and JSON filenames
# train -> base.json (for training)
# valid -> val.json (for validation)  
# test -> novel.json (for testing)
split_mapping = {
    'train': 'base',
    'valid': 'val',
    'test': 'novel'
}

# Get all class names across all splits for consistent label mapping
all_classes = set()
for split_dir in ['train', 'valid', 'test']:
    data_path = join(cwd, split_dir)
    if isdir(data_path):
        folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
        all_classes.update(folder_list)

# Sort for consistent ordering
all_folder_list = sorted(list(all_classes))
label_dict = dict(zip(all_folder_list, range(0, len(all_folder_list))))

print(f"Total number of classes: {len(all_folder_list)}")
print("Classes:", all_folder_list)

# Process each split
for split_dir, json_name in split_mapping.items():
    data_path = join(cwd, split_dir)
    if not isdir(data_path):
        print(f"Warning: {split_dir} directory not found, skipping...")
        continue
    
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()
    
    file_list = []
    label_list = []
    
    for folder in folder_list:
        folder_path = join(data_path, folder)
        classfile_list = [join(folder_path, cf) for cf in listdir(folder_path) 
                          if isfile(join(folder_path, cf)) and cf[0] != '.' 
                          and cf.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        
        if len(classfile_list) == 0:
            print(f"Warning: No images found in {folder_path}")
            continue
            
        file_list.extend(classfile_list)
        label_list.extend(np.repeat(label_dict[folder], len(classfile_list)).tolist())
    
    # Write JSON file
    output_file = join(cwd, json_name + ".json")
    fo = open(output_file, "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in all_folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')
    
    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')
    
    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')
    
    fo.close()
    print(f"{json_name}.json - OK ({len(file_list)} images from {len(folder_list)} classes)")
