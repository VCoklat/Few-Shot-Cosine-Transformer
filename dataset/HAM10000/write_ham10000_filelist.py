"""
HAM10000 Dataset Filelist Generator
====================================
This script generates the base.json, val.json, and novel.json files 
for few-shot learning on the HAM10000 skin lesion dataset.

HAM10000 Dataset Structure:
- 7 skin lesion categories:
  - akiec: Actinic keratoses and intraepithelial carcinoma
  - bcc: Basal cell carcinoma  
  - bkl: Benign keratosis-like lesions
  - df: Dermatofibroma
  - mel: Melanoma
  - nv: Melanocytic nevi
  - vasc: Vascular lesions

Expected directory structure after download:
HAM10000/
├── Dataset/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
├── write_ham10000_filelist.py
└── download.txt

Split strategy (7 classes):
- Base (training): 3 classes (akiec, bcc, bkl) - indices 0, 1, 2
- Val (validation): 2 classes (df, mel) - indices 3, 4
- Novel (testing): 2 classes (nv, vasc) - indices 5, 6
"""

import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = os.getcwd()
data_path = join(cwd, 'Dataset')
savedir = './'
dataset_list = ['base', 'val', 'novel']

# Get all class folders
folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list, range(0, len(folder_list))))

print(f"Found {len(folder_list)} classes: {folder_list}")

# Split classes for few-shot learning
# HAM10000 has 7 classes, we split them as:
# Base: first 3 classes, Val: next 2 classes, Novel: last 2 classes
base_classes = set(folder_list[:3])
val_classes = set(folder_list[3:5])
novel_classes = set(folder_list[5:])

print(f"Base classes: {base_classes}")
print(f"Val classes: {val_classes}")
print(f"Novel classes: {novel_classes}")

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append([
        join(folder_path, cf) for cf in listdir(folder_path) 
        if (isfile(join(folder_path, cf)) and cf[0] != '.')
    ])
    random.shuffle(classfile_list_all[i])

for dataset in dataset_list:
    file_list = []
    label_list = []
    
    for i, classfile_list in enumerate(classfile_list_all):
        folder_name = folder_list[i]
        
        if 'base' in dataset and folder_name in base_classes:
            file_list = file_list + classfile_list
            label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset and folder_name in val_classes:
            file_list = file_list + classfile_list
            label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset and folder_name in novel_classes:
            file_list = file_list + classfile_list
            label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in folder_list])
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
    print(f"{dataset} - OK ({len(file_list)} images)")
