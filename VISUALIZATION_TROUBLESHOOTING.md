# Visualization Troubleshooting Guide

## Quick Fix for Kaggle/Colab Environments

If you're experiencing matplotlib import errors in Kaggle or Google Colab, use the automated fix:

```bash
# Automated fix (recommended)
python fix_visualization_deps.py

# Verification only
python fix_visualization_deps.py --verify
```

## Common Issues and Solutions

### Issue 1: AttributeError: _ARRAY_API not found

**Symptom:**
```python
Traceback (most recent call last):
  File "feature_visualizer.py", line 7, in <module>
    import matplotlib
  File ".../matplotlib/__init__.py", line 129, in <module>
    from . import _api, _version, cbook, _docstring, rcsetup
  ...
  File ".../matplotlib/transforms.py", line 49, in <module>
    from matplotlib._path import (
AttributeError: _ARRAY_API not found

Warning: matplotlib import failed (numpy.core.multiarray failed to import).
```

**Root Cause:**
- matplotlib was installed/compiled against numpy 2.x
- Your environment now has numpy 1.x (as per requirements.txt)
- Binary incompatibility between matplotlib's compiled extensions and numpy

**Solution:**

**Option A: Automated Fix (Recommended)**
```bash
python fix_visualization_deps.py
```

**Option B: Manual Fix**
```bash
# Step 1: Ensure numpy 1.x is installed
pip install 'numpy>=1.23.0,<2.0.0'

# Step 2: Force reinstall matplotlib (this is the key step!)
pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5.0,<3.9.0'

# Step 3: Verify
python -c "import matplotlib; print('Success!', matplotlib.__version__)"
```

**Why this works:**
- `--force-reinstall` ensures matplotlib is reinstalled even if already present
- `--no-cache-dir` prevents using cached wheels that may be compiled with wrong numpy
- This gets matplotlib wheels compatible with your current numpy version

### Issue 2: Numba needs NumPy 1.26 or less

**Symptom:**
```
Warning: umap-learn import failed (Numba needs NumPy 1.26 or less)
UMAP visualization will be unavailable.
```

**Root Cause:**
- NumPy 2.x is installed
- numba (required by umap-learn) doesn't support numpy 2.x yet

**Solution:**
```bash
# Downgrade numpy to 1.x
pip install 'numpy>=1.23.0,<2.0.0'

# Reinstall numba and umap-learn
pip install --force-reinstall --no-cache-dir 'numba>=0.57.0,<0.60.0' 'umap-learn>=0.5.3'
```

### Issue 3: numpy.core.multiarray failed to import

**Symptom:**
```
ImportError: numpy.core.multiarray failed to import
```

**Root Cause:**
Same as Issue 1 - binary incompatibility between numpy and matplotlib.

**Solution:**
Follow the solution for Issue 1 above.

### Issue 4: Missing Visualization Dependencies

**Symptom:**
```
Warning: matplotlib import failed. Visualization features will be limited.
Warning: scikit-learn import failed. Some visualization features will be unavailable.
Error: matplotlib is required for visualization but could not be imported.
```

**Solution:**

For fresh installation:
```bash
pip install -r requirements.txt
```

For existing problematic installation:
```bash
# Use the fix script
python fix_visualization_deps.py

# Or manually:
pip install --force-reinstall --no-cache-dir -r requirements.txt
```

## Environment-Specific Instructions

### Kaggle Notebooks

**Problem:** Kaggle pre-installs packages with numpy 2.x, causing compatibility issues.

**Solution:**
```python
# In first cell of your Kaggle notebook
!pip install 'numpy>=1.23.0,<2.0.0'
!pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5.0,<3.9.0' 'numba>=0.57.0,<0.60.0' 'umap-learn>=0.5.3' seaborn pandas plotly scikit-learn

# Restart the kernel after installation (Kernel → Restart Kernel)
```

Or use the fix script:
```python
# After cloning the repository
!cd Few-Shot-Cosine-Transformer && python fix_visualization_deps.py

# Then restart kernel
```

### Google Colab

Similar to Kaggle, but usually has fewer issues:
```python
!pip install 'numpy>=1.23.0,<2.0.0'
!pip install --force-reinstall --no-cache-dir matplotlib

# Restart runtime after installation (Runtime → Restart runtime)
```

### Local Development (Ubuntu/Mac/Windows)

**Best Practice:** Use a virtual environment:

```bash
# Create virtual environment
python -m venv fsct_env

# Activate it
source fsct_env/bin/activate  # Linux/Mac
# OR
fsct_env\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt

# Verify
python fix_visualization_deps.py --verify
```

## Understanding the Error

### Why does this happen?

1. **Binary Wheel Compilation:** matplotlib includes C extensions compiled against specific numpy versions
2. **NumPy ABI Changes:** NumPy 2.0 introduced breaking changes in the binary interface
3. **Pip Caching:** pip may reuse cached wheels compiled with the wrong numpy version

### Prevention

To prevent these issues in the future:

1. Always install numpy first, before other scientific packages
2. Use `--no-cache-dir` when reinstalling after numpy version changes
3. Create fresh virtual environments for new projects
4. Pin all dependency versions in production (use `pip freeze`)

## Graceful Degradation

The visualization module is designed to degrade gracefully:

1. **No matplotlib**: Static plots unavailable, interactive plotly visualizations may work
2. **No scikit-learn**: PCA and t-SNE unavailable
3. **No UMAP**: UMAP skipped, PCA and t-SNE work
4. **No plotly**: Interactive visualizations unavailable, static matplotlib plots work

The code will inform you about available/missing features and allow training/evaluation to continue.

## Testing Your Installation

Run the test scripts to verify:

```bash
# Comprehensive test
python test_numpy_matplotlib_fix.py

# Quick verification
python fix_visualization_deps.py --verify

# Test feature visualizer specifically
python -c "import feature_visualizer; print('✓ Success!')"
```

Expected output:
```
✓ ALL TESTS PASSED!
The NumPy/Matplotlib compatibility issue has been fixed.
```

## Still Having Issues?

If problems persist:

1. **Check Python version:** Python 3.7-3.11 recommended (3.12+ may have compatibility issues)
   ```bash
   python --version
   ```

2. **Check installed versions:**
   ```bash
   pip list | grep -E "(numpy|matplotlib|numba|umap)"
   ```

3. **Clear all caches and reinstall:**
   ```bash
   pip cache purge
   pip uninstall numpy matplotlib numba umap-learn scikit-learn -y
   pip install --no-cache-dir -r requirements.txt
   ```

4. **Create a fresh virtual environment:**
   ```bash
   deactivate  # if in a venv
   rm -rf fsct_env  # or your venv name
   python -m venv fsct_env_new
   source fsct_env_new/bin/activate
   pip install -r requirements.txt
   ```

5. **Report the issue:**
   - Include Python version
   - Include output of `pip list`
   - Include full error traceback
   - Mention your environment (Kaggle/Colab/Local/etc.)

## Getting Help

Contact: `quanghuy0497@gmail.com` with:
- Your environment details
- Full error message
- Output of `python fix_visualization_deps.py --verify`
