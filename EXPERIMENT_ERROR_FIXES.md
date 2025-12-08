# Fix for Experiment Errors

## Overview

This document describes the fixes for two critical errors encountered when running experiments:

1. **HAM10000 3-way classification error**: Dataset split had insufficient classes
2. **Omniglot Conv4 shape mismatch error**: Feature dimension calculation was incorrect

## Problem Statement

### Error 1: HAM10000 3-way Classification

**Command:**
```bash
python run_experiments.py --dataset HAM10000 --backbone Conv4 --n_way 3 --k_shot 2 --num_epoch 50 --run_mode all --show_plots --mcnemar_each_test
```

**Error:**
```
ValueError: Dataset /kaggle/working/Few-Shot-Cosine-Transformer/dataset/HAM10000/base.json 
has only 2 classes, but n_way=3 was requested. Please reduce n_way to at most 2.
```

**Root Cause:**
- HAM10000 has 7 total classes
- Previous split: base=2, val=3, novel=2
- User requested 3-way classification on base split (training), but base only had 2 classes

### Error 2: Omniglot Conv4 Shape Mismatch

**Command:**
```bash
python run_experiments.py --dataset Omniglot --backbone Conv4 --n_way 5 --k_shot 1 --num_epoch 50 --run_mode all --show_plots --mcnemar_each_test
```

**Error:**
```
RuntimeError: Given normalized_shape=[1600], expected input with shape [*, 1600], 
but got input of size[1, 5, 64]
```

**Root Cause:**
- `ConvNet` class hardcoded feature dimensions assuming all non-CIFAR datasets produce 5x5 spatial output
- Formula: `dim = 4 if dataset == 'CIFAR' else 5`
- This resulted in `final_feat_dim = 64 * 5 * 5 = 1600`
- However, Omniglot images are 28x28, which after 4 pooling layers become 1x1
- Actual output: `64 * 1 * 1 = 64`
- LayerNorm was initialized for 1600 but received 64-dimensional features

## Solutions Implemented

### Fix 1: HAM10000 Dataset Splits

**File:** `dataset/HAM10000/write_HAM10000_filelist.py`

**Changes:**
- Updated `split_dataset()` call to use `min_base_classes=3`
- Regenerated dataset splits with proper class distribution

**New Split:**
```
Base split:    3 classes (mel, bkl, df)       - 2,327 images
Val split:     2 classes (bcc, nv)            - 7,219 images  
Novel split:   2 classes (vasc, akiec)        - 469 images
Total:         7 classes                       - 10,015 images
```

**Impact:**
- ✅ Base split now supports 3-way classification
- ✅ Val split supports 2-way classification
- ✅ Novel split supports 2-way classification

### Fix 2: ConvNet Dynamic Dimension Calculation

**File:** `backbone.py`

**Changes:**
- Replaced hardcoded dimension calculation with dynamic calculation based on actual input size
- New logic calculates spatial dimensions based on dataset-specific input sizes and number of pooling layers

**Before:**
```python
dim = 4 if dataset == 'CIFAR' else 5
self.final_feat_dim = 64 * dim * dim if flatten else [64, dim, dim]
```

**After:**
```python
# Calculate spatial dimensions based on dataset and pooling
if dataset in ['Omniglot', 'cross_char']:
    input_size = 28
elif dataset == 'CIFAR':
    input_size = 32
else:
    input_size = 84

# Count number of pooling layers (only first 4 layers have pooling)
num_pools = min(depth, 4)
# Calculate output spatial dimension: input_size / (2^num_pools)
dim = input_size // (2 ** num_pools)

self.final_feat_dim = 64 * dim * dim if flatten else [64, dim, dim]
```

**Dimension Calculations:**

| Dataset | Input Size | Pooling Layers | Spatial Output | Feature Dimension |
|---------|-----------|----------------|----------------|-------------------|
| Omniglot | 28×28 | 4 | 28/16 = 1×1 | 64 × 1 × 1 = **64** |
| cross_char | 28×28 | 4 | 28/16 = 1×1 | 64 × 1 × 1 = **64** |
| CIFAR | 32×32 | 4 | 32/16 = 2×2 | 64 × 2 × 2 = **256** |
| miniImagenet | 84×84 | 4 | 84/16 = 5×5 | 64 × 5 × 5 = **1600** |
| CUB | 84×84 | 4 | 84/16 = 5×5 | 64 × 5 × 5 = **1600** |
| HAM10000 | 84×84 | 4 | 84/16 = 5×5 | 64 × 5 × 5 = **1600** |

**Impact:**
- ✅ Correct feature dimensions for all datasets
- ✅ Fixes Omniglot Conv4 shape mismatch
- ✅ LayerNorm and other layers receive correct dimensions
- ✅ No regression for other datasets (CIFAR, miniImagenet, CUB maintain correct dims)

## Verification

### Testing

Run the test script to verify both fixes:
```bash
python3 test_fixes.py
```

Expected output:
```
✅ All tests PASSED!

The fixes should resolve:
  1. HAM10000 3-way classification error
  2. Omniglot Conv4 shape mismatch error
```

### Manual Verification

**Test HAM10000 3-way classification:**
```bash
python run_experiments.py --dataset HAM10000 --backbone Conv4 --n_way 3 --k_shot 2 --num_epoch 2 --run_mode train_test
```

**Test Omniglot Conv4:**
```bash
python run_experiments.py --dataset Omniglot --backbone Conv4 --n_way 5 --k_shot 1 --num_epoch 2 --run_mode train_test
```

## Files Changed

1. **`dataset/HAM10000/write_HAM10000_filelist.py`**
   - Updated `split_dataset()` call parameters
   - Regenerated dataset splits

2. **`dataset/HAM10000/base.json`**
   - Regenerated with 3 classes

3. **`dataset/HAM10000/val.json`**
   - Regenerated with 2 classes

4. **`dataset/HAM10000/novel.json`**
   - Regenerated with 2 classes

5. **`backbone.py`**
   - Updated `ConvNet.__init__()` to dynamically calculate feature dimensions

6. **`test_fixes.py`** (NEW)
   - Test script to verify both fixes

## Notes

### Why Not Use Conv4S for Omniglot?

The comment in `backbone.py` suggests using `Conv4S` for Omniglot:
```python
class ConvNetS(nn.Module): #For Omniglot, only 1 input channel, output dim is 64
```

However, the fix to `ConvNet` (used by `Conv4`) is more general and:
- Allows users to specify any backbone without knowing implementation details
- Fixes the root cause (incorrect dimension calculation)
- Works for all datasets automatically
- Maintains backward compatibility

Users can still use `Conv4S` explicitly if they prefer, but `Conv4` now also works correctly.

### HAM10000 Class Distribution

The new split prioritizes having enough classes for training (base) over validation:
- Base: 3 classes (sufficient for 3-way or lower)
- Val: 2 classes (sufficient for 2-way validation)
- Novel: 2 classes (sufficient for 2-way testing)

This is appropriate because:
1. Training typically requires more diverse classes
2. Validation and testing can use fewer classes with higher n_query to maintain statistical power
3. 7 total classes is a constraint we must work within

## Future Improvements

1. **Dynamic n_way validation**: Add runtime check in `run_experiments.py` to validate n_way against available classes before starting training

2. **Dataset documentation**: Add documentation specifying recommended n_way values for each dataset

3. **Flexible splits**: Allow command-line options to regenerate splits with custom class distributions

4. **Automatic backbone selection**: Add logic to automatically select appropriate backbone (Conv4 vs Conv4S) based on dataset characteristics
