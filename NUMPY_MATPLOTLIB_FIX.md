# NumPy/Matplotlib Compatibility Fix

## Problem

The repository was experiencing compatibility issues when numpy 2.x was installed:

1. **Matplotlib Import Error**: `AttributeError: _ARRAY_API not found`
   - This occurs because matplotlib (compiled with numpy 1.x) is incompatible with numpy 2.x
   - The error happens when importing `matplotlib.transforms` which tries to access `_ARRAY_API`

2. **UMAP Import Error**: `Numba needs NumPy 2.0 or less. Got NumPy 2.3.`
   - umap-learn depends on numba, which requires numpy < 2.0
   - Installing numpy 2.x breaks umap-learn's functionality

## Root Cause

The requirements.txt file already specified `numpy>=1.23.0,<2.0.0`, but:
- Some package managers or environments may have installed numpy 2.x anyway
- numba wasn't explicitly pinned, allowing incompatible versions
- matplotlib didn't have an upper bound, potentially allowing incompatible versions

## Solution

Updated `requirements.txt` to explicitly pin compatible versions:

```
numpy>=1.23.0,<2.0.0        # Keep numpy 1.x (already specified)
numba>=0.57.0,<0.60.0        # Explicitly pin numba for numpy 1.x compatibility
matplotlib>=3.5.0,<3.9.0     # Add upper bound for compatibility
umap-learn>=0.5.3            # Specify minimum version
```

### Why These Versions?

- **numpy < 2.0**: Required by both matplotlib and numba
- **numba 0.57-0.59**: Latest versions compatible with numpy 1.x
- **matplotlib < 3.9.0**: Ensures compatibility with numpy 1.x
- **umap-learn >= 0.5.3**: Stable version with good numba integration

## Verification

Three test scripts verify the fix:

1. **test_numpy_matplotlib_fix.py** (new): Comprehensive test for all imports
2. **test_numpy_compatibility.py** (existing): Tests numpy version constraints
3. **test_visualization_import.py** (existing): Tests feature_visualizer module

Run any of these to verify the fix:

```bash
python test_numpy_matplotlib_fix.py
```

Expected output:
```
✓ ALL TESTS PASSED!
The NumPy/Matplotlib compatibility issue has been fixed.
```

## Installation

After this fix, install dependencies with:

```bash
pip install -r requirements.txt
```

This will ensure:
- numpy 1.x is installed (not 2.x)
- Compatible versions of matplotlib, numba, and umap-learn are installed
- All visualization features work correctly

## Impact

- ✅ Fixes matplotlib import errors
- ✅ Fixes umap-learn import errors
- ✅ Maintains backward compatibility
- ✅ All existing functionality preserved
- ✅ No changes to source code required
