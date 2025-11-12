# HAM10000 Dataset Integration

This directory contains the implementation for using the HAM10000 skin lesion dataset with the Few-Shot-Cosine-Transformer framework.

## About HAM10000 Dataset

The HAM10000 (Human Against Machine with 10000 training images) dataset is a large collection of multi-source dermatoscopic images of pigmented lesions. It contains **10,015 dermatoscopic images** categorized into **7 diagnostic categories**:

1. **akiec**: Actinic keratoses and intraepithelial carcinoma
2. **bcc**: Basal cell carcinoma
3. **bkl**: Benign keratosis-like lesions
4. **df**: Dermatofibroma
5. **mel**: Melanoma
6. **nv**: Melanocytic nevi
7. **vasc**: Vascular lesions

## Dataset Download

Download the HAM10000 dataset from Kaggle:
- Dataset URL: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

You'll need:
1. `HAM10000_images_part_1.zip` - First part of images
2. `HAM10000_images_part_2.zip` - Second part of images
3. `HAM10000_metadata.csv` - Metadata file with image IDs and labels

## Setup Instructions

### 1. Extract the Dataset

```bash
cd dataset/HAM10000/

# Extract image files
unzip HAM10000_images_part_1.zip
unzip HAM10000_images_part_2.zip

# Move metadata file here
cp /path/to/HAM10000_metadata.csv .
```

Your directory structure should look like:
```
dataset/HAM10000/
├── HAM10000_images_part_1/
│   └── *.jpg (images)
├── HAM10000_images_part_2/
│   └── *.jpg (images)
├── HAM10000_metadata.csv
├── write_ham10000_filelist.py
└── README.md
```

### 2. Generate JSON Metadata Files

```bash
cd dataset/HAM10000/
python write_ham10000_filelist.py
```

This will create three JSON files:
- `base.json` - Training set (4 classes)
- `val.json` - Validation set (2 classes)
- `novel.json` - Test set (1 class)

## Usage

### Training

```bash
python train.py \
    --dataset HAM10000 \
    --method FSCT_cosine \
    --backbone ResNet18 \
    --n_way 5 \
    --k_shot 5 \
    --train_aug 1
```

### Testing

```bash
python test.py \
    --dataset HAM10000 \
    --method FSCT_cosine \
    --backbone ResNet18 \
    --n_way 5 \
    --k_shot 5
```

## Dataset Class API

The `HAM10000Dataset` class provides a PyTorch-compatible interface:

```python
from data.ham10000_dataset import HAM10000Dataset
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = HAM10000Dataset(
    csv_file='dataset/HAM10000/HAM10000_metadata.csv',
    img_dirs=[
        'dataset/HAM10000/HAM10000_images_part_1',
        'dataset/HAM10000/HAM10000_images_part_2'
    ],
    transform=transform,
    device='cuda'
)

# Get a sample
image, label = dataset[0]

# Get label mapping
label_mapping = dataset.get_label_mapping()
print(f"Label {label}: {label_mapping[label]}")

# Get class distribution
class_counts = dataset.get_class_counts()
print(class_counts)
```

## Class Distribution

The HAM10000 dataset has an imbalanced class distribution:
- **nv** (Melanocytic nevi): ~6,705 images (67%)
- **mel** (Melanoma): ~1,113 images (11%)
- **bkl** (Benign keratosis): ~1,099 images (11%)
- **bcc** (Basal cell carcinoma): ~514 images (5%)
- **akiec** (Actinic keratoses): ~327 images (3%)
- **vasc** (Vascular lesions): ~142 images (1%)
- **df** (Dermatofibroma): ~115 images (1%)

This makes it an interesting dataset for few-shot learning, especially for rare classes.

## Few-Shot Learning Considerations

### Why HAM10000 is Suitable for Few-Shot Learning

1. **Medical Domain**: Real-world medical diagnosis often has limited labeled data
2. **Class Imbalance**: Some lesion types are rare, making few-shot learning valuable
3. **Visual Similarity**: Different lesion types can look similar, requiring fine-grained discrimination
4. **Clinical Relevance**: Improving diagnosis with limited data has practical applications

### Recommended Settings

For HAM10000 dataset, we recommend:
- **5-way 1-shot**: Very challenging, tests true few-shot capability
- **5-way 5-shot**: Standard few-shot setting
- **Image size**: 224x224 (matches medical image standards)
- **Backbone**: ResNet18 or ResNet34 (good balance of capacity and efficiency)
- **Augmentation**: Use during training (horizontal flip, color jitter)

## Dataset Split Strategy

The `write_ham10000_filelist.py` script splits the 7 classes as follows:
- **Base** (4 classes): Used for meta-training
- **Val** (2 classes): Used for validation during training
- **Novel** (1 class): Used for final evaluation on unseen classes

This ensures the model is tested on truly novel lesion types.

## Citation

If you use the HAM10000 dataset, please cite:

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

## Troubleshooting

### Image Not Found Errors
- Ensure both image directories are extracted properly
- Check that image filenames in CSV match actual files
- Images should have `.jpg` extension

### Memory Issues
- Reduce batch size or n_query
- Use smaller backbone (Conv4 instead of ResNet)
- Enable gradient checkpointing

### Class Imbalance
- The dataset is naturally imbalanced
- Consider oversampling rare classes during meta-training
- Use class-balanced sampling strategies

## License

The HAM10000 dataset is licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
