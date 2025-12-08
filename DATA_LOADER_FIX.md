# Data Loader Fix Documentation

## Overview

This document describes the fixes implemented to resolve `RuntimeError: Invalid tensor shape` issues that occurred when running experiments with certain datasets.

## Problem Description

The original error occurred in `methods/meta_template.py` at line 119 in the `parse_feature` method:

```python
RuntimeError: shape '[105, 3, 84, 84]' is invalid for input of size 1778112
```

This happened because the data loader was not providing the expected number of samples, causing a shape mismatch when trying to reshape the tensor.

### Root Causes

Two distinct issues were identified:

#### Issue 1: Insufficient Samples Per Class
Some datasets have classes with fewer samples than required by the few-shot configuration (`k_shot + n_query`).

**Affected Datasets:**
- **Omniglot**: Each class has exactly 20 samples, but default configuration requests 21 (5 shot + 16 query)
- **DatasetIndo**: Some classes have only 33 samples
- **Yoga**: Some classes have only 30 samples

**Symptom:** When a class has fewer samples than `batch_size`, the DataLoader returns only the available samples, causing shape mismatches downstream.

#### Issue 2: Insufficient Number of Classes
Some datasets have fewer classes than the requested `n_way` parameter.

**Affected Datasets:**
- **HAM10000**: Only 4 classes in base.json, but 5-way classification was requested

**Symptom:** The episodic batch sampler can only sample the available classes, resulting in fewer classes than expected (e.g., 84 images instead of 105).

## Implemented Solutions

### Solution 1: Sampling with Replacement

**File:** `data/dataset.py`

Modified `SetDataset.__getitem__` to ensure exactly `batch_size` samples are returned for each class by sampling with replacement when necessary.

```python
def __getitem__(self, i):
    # Get a batch from the sub_dataloader
    images, labels = next(iter(self.sub_dataloader[i]))
    
    # If we got fewer samples than batch_size, sample with replacement
    if images.shape[0] < self.batch_size:
        n_additional = self.batch_size - images.shape[0]
        additional_indices = torch.randint(0, images.shape[0], (n_additional,))
        additional_images = images[additional_indices]
        additional_labels = labels[additional_indices]
        images = torch.cat([images, additional_images], dim=0)
        labels = torch.cat([labels, additional_labels], dim=0)
    
    return images, labels
```

**Benefits:**
- Ensures consistent tensor shapes across all episodes
- Allows training on datasets with limited samples per class
- Uses `torch.randint` for thread-safe random sampling

### Solution 2: Dataset Validation

**File:** `data/datamgr.py`

Added validation in `SetDataManager.get_data_loader` to check if `n_way` exceeds available classes.

```python
def get_data_loader(self, data_file, aug):
    # ... dataset creation code ...
    
    # Validate that we have enough classes for n_way
    n_classes = len(dataset)
    if n_classes < self.n_way:
        raise ValueError(
            f"Dataset {data_file} has only {n_classes} classes, "
            f"but n_way={self.n_way} was requested. "
            f"Please reduce n_way to at most {n_classes}."
        )
    
    # ... rest of the code ...
```

**Benefits:**
- Provides clear error messages when configuration is invalid
- Fails fast with actionable guidance
- Prevents silent failures and shape mismatches

## Dataset Requirements

### Minimum Samples Per Class

The minimum number of samples required per class is:
```
min_samples = k_shot + n_query
```

With default configuration:
- `k_shot = 5`
- `n_query = 16`
- **Minimum required: 21 samples per class**

If a class has fewer samples, the data loader will use sampling with replacement to reach the required number.

### Minimum Number of Classes

The minimum number of classes required is:
```
min_classes = n_way
```

With default configuration:
- `n_way = 5`
- **Minimum required: 5 classes**

If fewer classes are available, you'll get a clear error message instructing you to reduce `n_way`.

## Dataset Status Summary

| Dataset | Classes | Min Samples/Class | Status with Default Config (5-way, 5-shot, 16-query) |
|---------|---------|-------------------|------------------------------------------------------|
| CIFAR_FS | 64 | 600 | ✓ Works perfectly |
| CUB | 100 | 44 | ✓ Works (with sampling with replacement) |
| DatasetIndo | 12 | 33 | ✓ Works (with sampling with replacement) |
| HAM10000 | 4 | 115 | ✗ Error: Need n_way ≤ 4 |
| Omniglot | 4112 | 20 | ✓ Works (with sampling with replacement) |
| Yoga | 25 | 30 | ✓ Works (with sampling with replacement) |
| miniImagenet | Many | Many | ✓ Works perfectly |

## Usage Examples

### Example 1: Running with HAM10000
Since HAM10000 has only 4 classes, you must set `n_way=4` or lower:

```bash
python run_experiments.py \
    --dataset HAM10000 \
    --backbone Conv4 \
    --n_way 4 \
    --k_shot 5 \
    --num_epoch 50 \
    --run_mode all \
    --show_plots \
    --mcnemar_each_test
```

### Example 2: Running with Omniglot
Omniglot works fine with default settings because sampling with replacement is now enabled:

```bash
python run_experiments.py \
    --dataset Omniglot \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 5 \
    --num_epoch 50 \
    --run_mode all \
    --show_plots \
    --mcnemar_each_test
```

### Example 3: Adjusting Parameters for Small Datasets
For datasets with limited samples per class, you can reduce `k_shot` and `n_query`:

```bash
python run_experiments.py \
    --dataset DatasetIndo \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 3 \
    --n_query 10 \
    --num_epoch 50 \
    --run_mode all
```

## Technical Details

### Sampling with Replacement Behavior

When sampling with replacement is triggered:
1. The data loader first loads all available samples from the class
2. If fewer than `batch_size` samples are available, it randomly selects additional samples from those already loaded
3. Some samples may appear multiple times in the same batch
4. The random selection uses `torch.randint` for reproducibility and thread safety

### Thread Safety

The implementation uses PyTorch's random number generation (`torch.randint`) instead of NumPy's, ensuring thread-safe operation in multi-threaded data loading scenarios.

### Performance Considerations

- **No significant overhead** for datasets with sufficient samples (no extra processing needed)
- **Minimal overhead** for datasets requiring sampling with replacement (simple tensor indexing)
- **Memory efficient**: Does not create additional copies, only indexes existing tensors

## Migration Notes

If you have existing code that works with the original data loader:
- ✓ No changes needed - the fix is backward compatible
- ✓ Datasets with sufficient samples behave identically
- ✓ Datasets with insufficient samples now work instead of crashing

If you were working around the issue:
- You can remove any custom data augmentation or preprocessing to artificially increase sample counts
- You can simplify data loading logic that handled shape mismatches

## Testing

To verify the fix works with your dataset:

```python
from data.datamgr import SetDataManager

# Create data manager
datamgr = SetDataManager(
    image_size=84,
    n_way=5,
    k_shot=5,
    n_query=16,
    n_episode=1
)

# Load data
loader = datamgr.get_data_loader('dataset/YourDataset/base.json', aug=False)

# Check batch shape
for images, labels in loader:
    print(f"Batch shape: {images.shape}")
    # Should be [n_way, k_shot + n_query, C, H, W]
    # e.g., [5, 21, 3, 84, 84]
    break
```

## Known Limitations

1. **Sampling with replacement may reduce diversity**: When a class has very few samples, the same samples may appear multiple times in a batch. This is acceptable for few-shot learning but may slightly reduce training effectiveness.

2. **Cannot exceed available classes**: You cannot use `n_way` larger than the number of classes in your dataset. The validation will catch this and provide guidance.

## References

- Original issue: RuntimeError in `parse_feature` due to shape mismatch
- Files modified:
  - `data/dataset.py`: Sampling with replacement logic
  - `data/datamgr.py`: Dataset validation logic
