# HAM10000 Dataset - Quick Start Guide

This guide provides a quick overview of using the HAM10000 skin lesion dataset with the Few-Shot-Cosine-Transformer framework.

## Quick Setup (3 Steps)

### 1. Download Dataset
```bash
# Download from Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# You need: HAM10000_images_part_1.zip, HAM10000_images_part_2.zip, HAM10000_metadata.csv
```

### 2. Extract & Setup
```bash
cd dataset/HAM10000/
unzip HAM10000_images_part_1.zip
unzip HAM10000_images_part_2.zip
cp /path/to/HAM10000_metadata.csv .
python write_ham10000_filelist.py
```

### 3. Run Training
```bash
cd ../..
python train.py --dataset HAM10000 --method FSCT_cosine --backbone ResNet18 --n_way 5 --k_shot 5 --train_aug 1
```

## Dataset Overview

- **Total Images**: 10,015 dermatoscopic images
- **Classes**: 7 diagnostic categories
- **Task**: Few-shot skin lesion classification
- **Split**: 4 base classes (train) + 2 val classes + 1 novel class (test)

### Diagnostic Categories

| Code  | Description                                    | Images | Percentage |
|-------|------------------------------------------------|--------|------------|
| nv    | Melanocytic nevi                               | 6,705  | 67%        |
| mel   | Melanoma                                       | 1,113  | 11%        |
| bkl   | Benign keratosis-like lesions                  | 1,099  | 11%        |
| bcc   | Basal cell carcinoma                           | 514    | 5%         |
| akiec | Actinic keratoses and intraepithelial carcinoma| 327    | 3%         |
| vasc  | Vascular lesions                               | 142    | 1%         |
| df    | Dermatofibroma                                 | 115    | 1%         |

## Command Examples

### Training

**Standard 5-way 5-shot:**
```bash
python train.py --dataset HAM10000 --method FSCT_cosine --backbone ResNet18 \
                --n_way 5 --k_shot 5 --train_aug 1 --num_epoch 50
```

**Challenging 5-way 1-shot:**
```bash
python train.py --dataset HAM10000 --method FSCT_cosine --backbone ResNet18 \
                --n_way 5 --k_shot 1 --train_aug 1 --num_epoch 50
```

**With ProFONet (better performance):**
```bash
python train.py --dataset HAM10000 --method FSCT_ProFONet --backbone ResNet18 \
                --n_way 5 --k_shot 5 --train_aug 1 --num_epoch 50
```

### Testing

```bash
python test.py --dataset HAM10000 --method FSCT_cosine --backbone ResNet18 \
               --n_way 5 --k_shot 5
```

## Python API Usage

### Basic Usage

```python
from data.ham10000_dataset import HAM10000Dataset
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = HAM10000Dataset(
    csv_file='dataset/HAM10000/HAM10000_metadata.csv',
    img_dirs=['dataset/HAM10000/HAM10000_images_part_1',
              'dataset/HAM10000/HAM10000_images_part_2'],
    transform=transform,
    device='cuda'
)

# Access samples
image, label = dataset[0]
print(f"Image shape: {image.shape}, Label: {label}")

# Get label mapping
label_mapping = dataset.get_label_mapping()
print(label_mapping)  # {0: 'akiec', 1: 'bcc', ...}

# Get class distribution
class_counts = dataset.get_class_counts()
print(class_counts)
```

### With DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for images, labels in loader:
    # images: [batch_size, 3, 224, 224]
    # labels: [batch_size]
    pass
```

## Testing the Implementation

### Run the test script:
```bash
python test_ham10000_dataset.py
```

### Run the example:
```bash
python example_ham10000.py
```

## Expected Performance

Based on the dataset characteristics:

- **5-way 5-shot**: ~45-60% accuracy (baseline)
- **5-way 1-shot**: ~30-45% accuracy (challenging)
- **With ProFONet**: +5-10% improvement expected

Note: Actual performance depends on the specific class split and training setup.

## Tips for Best Results

1. **Use Data Augmentation**: `--train_aug 1` is recommended
2. **Choose Right Backbone**: 
   - ResNet18: Good balance, faster training
   - ResNet34: Better capacity, slower
   - Conv4: Fast but lower accuracy
3. **Monitor Validation**: Check val accuracy to avoid overfitting
4. **Class Imbalance**: The dataset is imbalanced; consider this in evaluation
5. **Medical Context**: Some classes are visually similar; expect lower accuracy than natural images

## Troubleshooting

### "CSV file not found"
- Download `HAM10000_metadata.csv` from Kaggle
- Place it in `dataset/HAM10000/`

### "Image not found" errors
- Extract both `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip`
- Images should be in `dataset/HAM10000/HAM10000_images_part_1/` and `part_2/`

### Memory issues
- Reduce `--n_query` (default is 16, try 8)
- Use smaller backbone (Conv4 instead of ResNet)
- Reduce batch size in data loader

### JSON files not found
- Run `python write_ham10000_filelist.py` in `dataset/HAM10000/`
- This creates `base.json`, `val.json`, and `novel.json`

## Files Structure

After setup, your directory should look like:

```
dataset/HAM10000/
├── HAM10000_images_part_1/
│   └── ISIC_*.jpg (5000+ images)
├── HAM10000_images_part_2/
│   └── ISIC_*.jpg (5000+ images)
├── HAM10000_metadata.csv
├── base.json (generated)
├── val.json (generated)
├── novel.json (generated)
├── write_ham10000_filelist.py
└── README.md
```

## Additional Resources

- **Full Documentation**: `dataset/HAM10000/README.md`
- **Test Script**: `test_ham10000_dataset.py`
- **Example Usage**: `example_ham10000.py`
- **Dataset Source**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Paper**: Tschandl et al., "The HAM10000 dataset", Scientific Data 2018

## Citation

If you use the HAM10000 dataset in your research, please cite:

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

## License

The HAM10000 dataset is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).
