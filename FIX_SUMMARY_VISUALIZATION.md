# Fix Summary: AttributeError: _ARRAY_API not found

## Issue Description

The original problem occurred in Kaggle environments where importing matplotlib resulted in:

```
AttributeError: _ARRAY_API not found
Error: feature_visualizer module not found
```

This happened when `train_test.py` tried to call visualization features through `eval_utils.py`, which in turn imported `feature_visualizer.py`. The import chain failed at the matplotlib import step.

## Root Cause

The error `AttributeError: _ARRAY_API not found` occurs when:
1. Matplotlib tries to import internal modules during initialization
2. The `matplotlib._path` module looks for the `_ARRAY_API` attribute
3. Due to environment incompatibilities (often in cloud platforms), this attribute is not found
4. The entire import chain crashes, preventing the program from running

## Solution Implemented

### 1. Graceful Import Handling (feature_visualizer.py)

**Before:**
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# ... other imports
```

**After:**
```python
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Warning: matplotlib import failed ({e}). Visualization features will be limited.")
    MATPLOTLIB_AVAILABLE = False
    plt = None
    Axes3D = None
    sns = None
```

### 2. Dependency Availability Flags

Added flags to track which dependencies are available:
- `MATPLOTLIB_AVAILABLE`
- `SKLEARN_AVAILABLE`
- `UMAP_AVAILABLE`
- `TORCH_AVAILABLE`
- `PLOTLY_AVAILABLE`
- `PANDAS_AVAILABLE`

### 3. Runtime Checks

Added checks before using each dependency:

```python
def visualize(self, ...):
    if not MATPLOTLIB_AVAILABLE or plt is None:
        raise ImportError("matplotlib is required for static visualization but could not be imported. "
                        "Try installing matplotlib with: pip install matplotlib>=3.5.0")
    # ... rest of the code
```

### 4. Enhanced Error Messages

Provides clear, actionable error messages:
- Identifies which dependency is missing
- Suggests installation commands
- Returns None gracefully instead of crashing

### 5. Updated Error Handling in eval_utils.py

**Before:**
```python
try:
    from feature_visualizer import visualize_features_from_results
except ImportError:
    print("Error: feature_visualizer module not found")
    return None
```

**After:**
```python
try:
    from feature_visualizer import visualize_features_from_results
except (ImportError, AttributeError) as e:
    print(f"Error: Could not import visualization module: {e}")
    print("Visualization features require matplotlib, scikit-learn, and other dependencies.")
    print("Please ensure all requirements are installed: pip install -r requirements.txt")
    return None
```

## Key Improvements

1. **No More Crashes**: The code continues running even if matplotlib fails to import
2. **Clear Diagnostics**: Users see exactly which dependencies are missing
3. **Graceful Degradation**: Functions return None instead of raising exceptions
4. **Helpful Messages**: Installation instructions are provided automatically
5. **Backend Selection**: Automatically uses 'Agg' backend to avoid display issues

## Testing

### Test 1: Basic Import (test_visualization_import.py)
```
✓ Module imports successfully even when matplotlib is not available
✓ Provides clear warning messages about missing dependencies
✓ Returns None gracefully when dependencies are unavailable
```

### Test 2: Kaggle Error Simulation (test_kaggle_error_fix.py)
```
✓ Module imports without AttributeError: _ARRAY_API not found
✓ Provides clear warning messages about missing dependencies
✓ Code continues to run instead of crashing
✓ Returns None gracefully when visualization is not possible
```

## Files Modified

1. **feature_visualizer.py** (202 lines changed)
   - Wrapped all imports in try-except blocks
   - Added availability flags
   - Added runtime checks in all methods

2. **eval_utils.py** (25 lines changed)
   - Enhanced error handling
   - Added detailed error messages

3. **test_visualization_import.py** (121 lines, new file)
   - Tests basic import handling

4. **test_kaggle_error_fix.py** (136 lines, new file)
   - Simulates the exact error scenario

5. **VISUALIZATION_TROUBLESHOOTING.md** (131 lines, new file)
   - Comprehensive troubleshooting guide

## Backward Compatibility

The changes are fully backward compatible:
- When all dependencies are available, code works exactly as before
- When dependencies are missing, code degrades gracefully
- No changes to API or function signatures
- Existing code that doesn't use visualization features is unaffected

## Usage Examples

### Example 1: Successful Visualization (all dependencies available)
```python
from eval_utils import visualize_feature_projections

result = visualize_feature_projections(
    test_loader, model, n_way=5, 
    device='cuda', show=True
)
# Works as before, generates visualizations
```

### Example 2: Missing Dependencies (matplotlib not available)
```python
from eval_utils import visualize_feature_projections

result = visualize_feature_projections(
    test_loader, model, n_way=5, 
    device='cuda', show=True
)
# Output:
# Warning: matplotlib import failed (AttributeError: _ARRAY_API not found)
# Error: matplotlib is required for visualization but could not be imported.
# Please install it with: pip install matplotlib>=3.5.0
# result = None (but program continues running)
```

## Deployment Notes

### For Kaggle Users
If you still encounter issues after this fix:
```python
# In a notebook cell
!pip install matplotlib==3.5.0
# Restart runtime
```

### For Google Colab Users
Usually works with default installation, but if needed:
```python
!pip install --upgrade matplotlib
```

### For Local Development
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Conclusion

This fix resolves the `AttributeError: _ARRAY_API not found` issue by:
1. Catching the error before it propagates
2. Providing clear diagnostic information
3. Allowing the program to continue running
4. Maintaining full backward compatibility

Users can now run the training and evaluation code even in environments where matplotlib has compatibility issues, with clear guidance on how to enable visualization features if needed.
