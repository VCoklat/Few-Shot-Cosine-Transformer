# HAM10000 Dataset Integration Guide

This guide explains how to use the HAM10000 skin cancer dataset with the Few-Shot Cosine Transformer framework, specifically addressing the code structure provided in the problem statement.

## Overview

The HAM10000 dataset contains over 10,000 dermatoscopic images of pigmented skin lesions across 7 classes:
- **akiec**: Actinic keratoses
- **bcc**: Basal cell carcinoma
- **bkl**: Benign keratosis-like lesions
- **df**: Dermatofibroma
- **mel**: Melanoma
- **nv**: Melanocytic nevi
- **vasc**: Vascular lesions

## Integration with Your Code

The code you provided from Kaggle has been integrated into this framework. Here's how it works:

### Your Original Code Structure
```python
# --- Path Configuration ---
dataset_base_path = '/kaggle/input/skin-cancer-the-ham10000-dataset/'
path_part1 = os.path.join(dataset_base_path, 'HAM10000_images_part_1')
path_part2 = os.path.join(dataset_base_path, 'HAM10000_images_part_2')
image_list_path = '/kaggle/input/d/rafiarrantisi/my-ham1000-final-list/final_1000_image_list.csv'

# --- Load Data ---
df_1000 = pd.read_csv(image_list_path)
```

### How It's Integrated

The `write_HAM10000_filelist.py` script uses the same structure and:
1. Reads your CSV file with image IDs and labels
2. Finds images in both `HAM10000_images_part_1` and `HAM10000_images_part_2`
3. Splits classes into train/val/test sets
4. Generates JSON files compatible with the framework

## Step-by-Step Setup

### 1. Download the HAM10000 Dataset

**Option A: From Kaggle**
```bash
# Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# Download and extract to get:
# - HAM10000_images_part_1/
# - HAM10000_images_part_2/
# - HAM10000_metadata.csv
```

**Option B: Using Kaggle API**
```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip
```

### 2. Prepare Your Image List CSV

Create a CSV file (e.g., `final_1000_image_list.csv`) with these columns:
```csv
image_id,dx
ISIC_0024306,bkl
ISIC_0025030,mel
ISIC_0027419,akiec
...
```

Where:
- `image_id`: Image filename without extension (e.g., ISIC_0024306)
- `dx`: Diagnosis/label (one of: akiec, bcc, bkl, df, mel, nv, vasc)

### 3. Configure Paths

Edit `dataset/HAM10000/write_HAM10000_filelist.py` and update these paths:

```python
# For Kaggle environment
dataset_base_path = '/kaggle/input/skin-cancer-the-ham10000-dataset/'
image_list_path = '/kaggle/input/d/rafiarrantisi/my-ham1000-final-list/final_1000_image_list.csv'

# For local environment
# dataset_base_path = './HAM10000_dataset/'
# image_list_path = './final_1000_image_list.csv'
```

### 4. Process the Dataset

```bash
cd dataset/HAM10000/
source HAM10000_processing.sh
```

This creates three JSON files:
- `base.json`: Training set (64% of classes)
- `val.json`: Validation set (16% of classes)
- `novel.json`: Test set (20% of classes)

### 5. Verify the Integration

```bash
cd ../..
python test_ham10000.py
```

Expected output:
```
============================================================
Testing HAM10000 Dataset Integration
============================================================
✓ Test 1: Checking configs.py...
  ✅ HAM10000 path: ./dataset/HAM10000/
...
✅ All tests passed!
```

## Training with HAM10000

### Basic Training
```bash
python train.py \
    --dataset HAM10000 \
    --method FSCT_cosine \
    --backbone ResNet34 \
    --n_way 5 \
    --k_shot 5 \
    --num_epoch 50
```

### With Data Augmentation (Recommended)
```bash
python train_test.py \
    --dataset HAM10000 \
    --method FSCT_cosine \
    --backbone ResNet34 \
    --n_way 5 \
    --k_shot 5 \
    --train_aug 1 \
    --wandb 1
```

### Using the Hybrid FSCT_ProFONet Method
```bash
python train.py \
    --method FSCT_ProFONet \
    --dataset HAM10000 \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 5 \
    --n_query 10 \
    --num_epoch 50
```

## Data Loading Behind the Scenes

When you run training, the framework:

1. **Loads JSON files** from `configs.data_dir['HAM10000']`
2. **Creates SetDataset** with your images:
   ```python
   dataset = SetDataset(data_file, batch_size, transform)
   ```
3. **Samples episodes** using EpisodicBatchSampler
4. **Applies transforms**:
   - Resize to target size (e.g., 84x84)
   - Optional augmentation (if `--train_aug 1`)
   - Normalize with ImageNet statistics

## Combining with Your Feature Extraction Code

If you want to use your GLCM features or other preprocessing from your original code:

### Option 1: Modify the Transform Pipeline
Edit `data/datamgr.py` to add custom transforms:
```python
class TransformLoader:
    def get_composed_transform(self, aug=False):
        transform_list = [
            'Resize', 
            'CenterCrop',
            YourGLCMTransform(),  # Add custom transform
            'ToTensor', 
            'Normalize'
        ]
        # ...
```

### Option 2: Preprocess and Save Features
Use your feature extraction code, then load features:
```python
# Extract features with your code
features = extract_glcm_features(images)
# Save for later use
h5py.File('ham10000_features.h5', 'w')
```

## Dataset Statistics

After processing, you'll have approximately:
- **Base set**: ~640 images from 5 classes (for training)
- **Val set**: ~160 images from 1 class (for validation)
- **Novel set**: ~200 images from 1 class (for testing)

These numbers will vary based on your CSV file and class distribution.

## Troubleshooting

### Issue: Images not found
**Solution**: Verify your paths and that images are in `HAM10000_images_part_1/` or `HAM10000_images_part_2/`

### Issue: CSV format error
**Solution**: Ensure CSV has columns `image_id` and `dx` (or `label`)

### Issue: Not enough classes for split
**Solution**: If you have fewer than 7 classes, adjust split ratios in `write_HAM10000_filelist.py`:
```python
splits = split_dataset(df, base_ratio=0.60, val_ratio=0.20, novel_ratio=0.20)
```

## Advanced: Custom Split Strategy

If you want a different split strategy, modify the `split_dataset` function:

```python
def custom_split(df):
    # Example: Split by patient ID instead of class
    patient_ids = df['patient_id'].unique()
    # Your custom logic here
    return {'base': base_df, 'val': val_df, 'novel': novel_df}
```

## Citation

If you use HAM10000, please cite:

```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific data},
  volume={5},
  number={1},
  pages={1--9},
  year={2018},
  publisher={Nature Publishing Group}
}
```

## Support

For questions about:
- **HAM10000 dataset**: See `download_HAM10000.txt`
- **Processing script**: Check `write_HAM10000_filelist.py` comments
- **Framework usage**: See main `README.md`

## Next Steps

1. ✅ Setup complete - Ready to train!
2. Run experiments with different hyperparameters
3. Compare with other datasets (miniImagenet, CUB, etc.)
4. Try different backbones and methods
5. Monitor training with WandB (`--wandb 1`)
