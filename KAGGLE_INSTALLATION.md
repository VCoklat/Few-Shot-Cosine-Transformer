# Installation Guide for Kaggle

This guide provides step-by-step instructions for installing and using this repository in Kaggle notebooks, addressing the common numpy/matplotlib compatibility issues.

## Quick Start (Recommended)

In your Kaggle notebook, run these commands in sequence:

### Step 1: Clone the Repository

```python
# In a Kaggle notebook cell
!git clone https://github.com/VCoklat/Few-Shot-Cosine-Transformer.git
%cd Few-Shot-Cosine-Transformer
```

### Step 2: Fix Dependencies

```python
# Run the automated fix
!python fix_visualization_deps.py

# Or use manual fix (if automated fix doesn't work)
!pip install 'numpy>=1.23.0,<2.0.0'
!pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5.0,<3.9.0' 'numba>=0.57.0,<0.60.0' 'umap-learn>=0.5.3' 'scikit-learn>=1.0.0' seaborn pandas plotly scipy
```

### Step 3: Restart the Kernel

**Important:** After installing dependencies, you MUST restart the Kaggle kernel:
- Click "Kernel" in the top menu
- Select "Restart Kernel" or "Restart & Clear Output"
- This ensures the new packages are properly loaded

### Step 4: Verify Installation

```python
# After restarting kernel, run this in a new cell
%cd Few-Shot-Cosine-Transformer
!python test_visualization_fix.py
```

Expected output:
```
✓ ALL CRITICAL IMPORTS SUCCESSFUL!
You can now use all visualization features.
```

### Step 5: Run Your Experiments

```python
# Example: Train and test with visualization
!python train_test.py --method FSCT_cosine --dataset miniImagenet --n_way 5 --k_shot 5 --visualize_features
```

## Understanding the Issue

### Why Do We Need This?

Kaggle pre-installs many Python packages, including:
- NumPy (often version 2.x)
- Matplotlib (compiled against the pre-installed NumPy)

When you install this project's requirements:
1. NumPy gets downgraded to 1.x (for compatibility with numba/umap)
2. But matplotlib's binary extensions are still linked to NumPy 2.x
3. This causes `AttributeError: _ARRAY_API not found` when importing matplotlib

### The Solution

The fix script does the following:
1. **Installs NumPy 1.x** - Ensures correct base version
2. **Force reinstalls matplotlib** - Rebuilds with correct NumPy linkage
3. **Reinstalls other viz dependencies** - Ensures all are compatible

The key is `--force-reinstall --no-cache-dir`, which:
- Forces package reinstallation even if already present
- Avoids using cached wheels that may be incompatible

## Common Errors and Solutions

### Error 1: _ARRAY_API not found

```python
AttributeError: _ARRAY_API not found
```

**Solution:**
```python
!pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5.0,<3.9.0'
# Then restart kernel
```

### Error 2: Numba needs NumPy 1.26 or less

```python
ImportError: Numba needs NumPy 1.26 or less
```

**Solution:**
```python
!pip install 'numpy>=1.23.0,<2.0.0'
!pip install --force-reinstall --no-cache-dir 'numba>=0.57.0,<0.60.0' 'umap-learn>=0.5.3'
# Then restart kernel
```

### Error 3: numpy.core.multiarray failed to import

```python
ImportError: numpy.core.multiarray failed to import
```

**Solution:**
This is the same as Error 1 - follow the matplotlib reinstallation steps above.

### Error 4: Visualization features unavailable

```python
Warning: matplotlib import failed. Visualization features will be limited.
```

**Solution:**
```python
!python fix_visualization_deps.py
# Then restart kernel
```

## Manual Installation (Alternative Method)

If the automated fix doesn't work, try this manual approach:

### Step 1: Install NumPy First

```python
!pip uninstall numpy -y
!pip install --no-cache-dir 'numpy>=1.23.0,<2.0.0'
```

### Step 2: Install Core Dependencies

```python
!pip install --no-cache-dir 'matplotlib>=3.5.0,<3.9.0' 'numba>=0.57.0,<0.60.0' 'scikit-learn>=1.0.0' scipy
```

### Step 3: Install Visualization Dependencies

```python
!pip install --no-cache-dir 'umap-learn>=0.5.3' seaborn pandas plotly
```

### Step 4: Install PyTorch and Other Requirements

```python
!pip install --no-cache-dir torch torchvision opencv-python h5py tqdm einops
```

### Step 5: Restart Kernel

Click "Kernel" → "Restart Kernel"

### Step 6: Verify

```python
!python test_visualization_fix.py
```

## Best Practices for Kaggle

### 1. Use Dedicated Notebook Sections

Organize your notebook into clear sections:

```python
# ===== SECTION 1: Setup =====
!git clone https://github.com/VCoklat/Few-Shot-Cosine-Transformer.git
%cd Few-Shot-Cosine-Transformer

# ===== SECTION 2: Install Dependencies =====
!python fix_visualization_deps.py
# [RESTART KERNEL HERE]

# ===== SECTION 3: Verify Installation =====
!python test_visualization_fix.py

# ===== SECTION 4: Run Experiments =====
# Your training/testing code here
```

### 2. Save Intermediate Results

Kaggle can disconnect or timeout. Save your work frequently:

```python
# Enable checkpointing
!python train_test.py --method FSCT_cosine --dataset miniImagenet --save_freq 10
```

### 3. Use Kaggle Datasets

For faster loading, consider uploading datasets to Kaggle Datasets and mounting them:

```python
# If you've uploaded the dataset to Kaggle
import os
os.symlink('/kaggle/input/mini-imagenet', '/kaggle/working/Few-Shot-Cosine-Transformer/data/miniImagenet')
```

### 4. Monitor Resource Usage

Kaggle has resource limits. Monitor them:

```python
import psutil
import GPUtil

# Check RAM
ram = psutil.virtual_memory()
print(f"RAM Usage: {ram.percent}%")

# Check GPU
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
```

## Troubleshooting

### Issue: Kernel Keeps Crashing

**Possible causes:**
- Out of memory (RAM or GPU)
- Incompatible package versions

**Solutions:**
1. Reduce batch size or episode count
2. Use gradient checkpointing: `--gradient_checkpointing 1`
3. Use Conv4 backbone instead of ResNet: `--backbone Conv4`

### Issue: "No space left on device"

**Solution:**
Clean up unnecessary files:

```python
# Remove downloaded datasets if not needed
!rm -rf /kaggle/working/data/downloaded_files

# Clear pip cache
!pip cache purge

# Remove temporary files
!rm -rf /tmp/*
```

### Issue: Import Errors After Kernel Restart

**Solution:**
Make sure to `%cd` back to the repository directory after restart:

```python
%cd /kaggle/working/Few-Shot-Cosine-Transformer
```

## Getting Help

If you still have issues:

1. **Check the logs:**
   ```python
   !cat record/results.txt
   ```

2. **Verify package versions:**
   ```python
   !pip list | grep -E "(numpy|matplotlib|numba|umap|torch)"
   ```

3. **Run full diagnostic:**
   ```python
   !python fix_visualization_deps.py --verify
   ```

4. **Contact support:**
   Email: quanghuy0497@gmail.com with:
   - Full error message/traceback
   - Output of verification script
   - Kaggle notebook link (if public)

## Example Kaggle Notebook Structure

Here's a complete example notebook structure:

```python
# Cell 1: Clone repository
!git clone https://github.com/VCoklat/Few-Shot-Cosine-Transformer.git
%cd Few-Shot-Cosine-Transformer

# Cell 2: Fix dependencies
!python fix_visualization_deps.py

# Cell 3: RESTART KERNEL HERE (do this manually)
# After restart, run the cells below

# Cell 4: Navigate back and verify
%cd /kaggle/working/Few-Shot-Cosine-Transformer
!python test_visualization_fix.py

# Cell 5: Download dataset (if needed)
# !cd data && bash download_miniImagenet.sh

# Cell 6: Train model
!python train_test.py \
    --method FSCT_cosine \
    --dataset miniImagenet \
    --backbone ResNet34 \
    --n_way 5 \
    --k_shot 5 \
    --num_epoch 50 \
    --visualize_features

# Cell 7: View results
!cat record/results.txt

# Cell 8: Generate visualizations
!python example_visualization.py
```

## Additional Resources

- [VISUALIZATION_TROUBLESHOOTING.md](VISUALIZATION_TROUBLESHOOTING.md) - Detailed troubleshooting guide
- [README.md](README.md) - Main documentation
- [OPTIMAL_FEW_SHOT.md](OPTIMAL_FEW_SHOT.md) - Information about the OptimalFewShot method
- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Guide to feature visualizations

## Success Indicators

You'll know everything is working when:

1. ✅ `test_visualization_fix.py` shows all green checkmarks
2. ✅ `feature_visualizer.MATPLOTLIB_AVAILABLE` is `True`
3. ✅ Training runs without import errors
4. ✅ Visualizations are generated successfully

Happy experimenting! 🎉
