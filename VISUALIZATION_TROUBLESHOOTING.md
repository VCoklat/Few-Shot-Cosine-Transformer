# Visualization Troubleshooting Guide

## Common Issues and Solutions

### AttributeError: _ARRAY_API not found

**Symptom:**
```
AttributeError: _ARRAY_API not found
Error: feature_visualizer module not found
```

**Cause:**
This error occurs when matplotlib has compatibility issues with your environment, often in cloud environments like Kaggle. It happens when matplotlib's internal modules cannot properly initialize.

**Solution:**
The codebase has been updated to handle this error gracefully. The visualization module will now:
1. Catch the error and continue running
2. Display a warning message about missing dependencies
3. Return `None` instead of crashing
4. Provide installation instructions

If you still encounter issues, try:

```bash
# Reinstall matplotlib with a specific version
pip uninstall matplotlib
pip install matplotlib==3.5.0

# Or try upgrading to the latest version
pip install --upgrade matplotlib

# Ensure numpy is compatible
pip install numpy>=1.23.0,<2.0.0
```

### Missing Visualization Dependencies

**Symptom:**
```
Warning: matplotlib import failed. Visualization features will be limited.
Warning: scikit-learn import failed. Some visualization features will be unavailable.
```

**Solution:**
Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install specific packages:

```bash
pip install matplotlib>=3.5.0
pip install scikit-learn>=1.0.0
pip install umap-learn
pip install seaborn
pip install plotly
pip install pandas
```

### Non-interactive Backend Issues

**Symptom:**
Plots don't display or you get display-related errors.

**Solution:**
The code now automatically sets matplotlib to use the 'Agg' backend (non-interactive), which works better in headless environments. If you need interactive plots in a Jupyter notebook, you can manually set the backend:

```python
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
```

## Graceful Degradation

The visualization module is designed to degrade gracefully:

1. **No matplotlib**: Static plots will not be available, but interactive plotly visualizations may still work
2. **No scikit-learn**: PCA and t-SNE visualizations will not be available
3. **No UMAP**: UMAP visualizations will be skipped, but PCA and t-SNE will still work
4. **No plotly**: Interactive visualizations will not be available, but static matplotlib plots will work

The code will inform you about what's available and what's missing, allowing you to continue training and evaluation even if some visualization features are unavailable.

## Testing Your Installation

Run the provided test scripts to verify your installation:

```bash
# Test basic import handling
python test_visualization_import.py

# Test the specific Kaggle error scenario
python test_kaggle_error_fix.py
```

Both tests should pass (or show clear warnings about missing dependencies) without crashing.

## Environment-Specific Notes

### Kaggle
In Kaggle notebooks, you may need to restart the runtime after installing packages:
```python
# In a Kaggle notebook cell
!pip install matplotlib==3.5.0
# Then restart the runtime from the menu
```

### Google Colab
Usually works well with the default matplotlib installation. If you encounter issues:
```python
!pip install --upgrade matplotlib
```

### Local Development
For local development, consider using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Getting Help

If you continue to experience issues:
1. Check which dependencies are available by examining the warning messages
2. Verify your Python version (3.7+ recommended)
3. Try creating a fresh virtual environment
4. Check for conflicting package versions with `pip list`
