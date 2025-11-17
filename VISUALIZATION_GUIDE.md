# Feature Visualization Guide

This guide explains how to use the PCA/t-SNE/UMAP projection visualization features added to the Few-Shot-Cosine-Transformer repository.

## Overview

The visualization feature allows you to visualize feature clustering after model evaluation using three dimensionality reduction techniques:
- **PCA (Principal Component Analysis)**: Linear projection preserving variance
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Non-linear projection preserving local structure
- **UMAP (Uniform Manifold Approximation and Projection)**: Non-linear projection balancing local and global structure

Each technique is applied in both **2D and 3D**, resulting in 6 visualizations displayed in a single figure.

## Quick Start

### 1. Using the Command-Line Interface

Add the `--visualize_features` flag to your test command:

```bash
python test.py --dataset miniImagenet --method FSCT_cosine \
    --backbone ResNet34 --n_way 5 --k_shot 5 \
    --visualize_features
```

This will:
1. Run the model evaluation
2. Extract features from the model
3. Generate all 6 visualizations (PCA/t-SNE/UMAP in 2D/3D)
4. Save the figure to `./figures/feature_projections/feature_projections_all.png`
5. Display the plots using `plt.show()` (if using an interactive backend)

### 2. Using the Python API

#### Option A: During Evaluation

```python
import eval_utils
from data.datamgr import SetDataManager

# Load your model and data
# ... (model loading code) ...

# Create data loader
test_datamgr = SetDataManager(image_size, n_episode=600, n_way=5, k_shot=5, n_query=15)
test_loader = test_datamgr.get_data_loader(testfile, aug=False)

# Visualize features
result = eval_utils.visualize_feature_projections(
    loader=test_loader,
    model=model,
    n_way=5,
    device='cuda',
    show=True,  # Display using plt.show()
    save_dir='./figures/my_visualizations'
)
```

#### Option B: With Pre-extracted Features

```python
from feature_visualizer import visualize_features_from_results
import numpy as np

# Assuming you have features and labels from your model
features = np.random.randn(100, 512)  # (n_samples, n_features)
labels = np.repeat(np.arange(5), 20)  # (n_samples,)

# Generate visualizations
result = visualize_features_from_results(
    features=features,
    labels=labels,
    show=True,  # Display using plt.show()
    save_dir='./figures/custom_visualizations',
    title_prefix="Custom Experiment"
)

# Access individual projections
pca_2d = result['projections']['PCA_2D']
tsne_3d = result['projections']['t-SNE_3D']
```

### 3. Running the Example Script

A standalone example is provided:

```bash
python example_visualization.py
```

This generates visualizations from synthetic data and saves them to `./figures/example_visualizations/`.

## Features

### Automatic Feature Extraction
The visualization system automatically extracts features from your model during evaluation. It supports models with:
- `parse_feature()` method (preferred for episodic models)
- `feature()` method (fallback)

### Multiple Projection Methods
All three dimensionality reduction techniques are computed and displayed:
- **PCA**: Fast, linear, preserves global structure
- **t-SNE**: Slower, non-linear, excellent for local structure
- **UMAP**: Balanced speed and quality, preserves both local and global structure

### 2D and 3D Visualizations
Each method is shown in both:
- **2D**: Better for printing and quick inspection
- **3D**: Better for understanding spatial relationships

### Interactive Display
When using an interactive backend (e.g., Qt5Agg, TkAgg):
- Plots are displayed using `plt.show()`
- 3D plots can be rotated and zoomed
- You can save individual plots interactively

### Automatic Saving
Visualizations are automatically saved as high-resolution PNG files (300 DPI) when `save_dir` is specified.

## Configuration

### Command-Line Arguments

```bash
--visualize_features     # Enable feature visualization (flag, no value needed)
--feature_analysis 1     # Also perform comprehensive feature analysis
--comprehensive_eval 1   # Use comprehensive evaluation metrics
```

### Matplotlib Backend

For **interactive display** (with `plt.show()`), ensure you have an appropriate backend:

```python
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg', or remove this line to use default
```

For **non-interactive** (save only):

```python
import matplotlib
matplotlib.use('Agg')
```

## Output

### File Structure
```
./figures/feature_projections/
└── feature_projections_all.png  # Combined 2x3 grid of all visualizations
```

### Return Values
The `visualize_features_from_results()` function returns a dictionary:

```python
{
    'projections': {
        'PCA_2D': ndarray,      # (n_samples, 2)
        'PCA_3D': ndarray,      # (n_samples, 3)
        't-SNE_2D': ndarray,    # (n_samples, 2)
        't-SNE_3D': ndarray,    # (n_samples, 3)
        'UMAP_2D': ndarray,     # (n_samples, 2)
        'UMAP_3D': ndarray      # (n_samples, 3)
    },
    'figure': matplotlib Figure object
}
```

## Examples

### Example 1: Basic Usage with Test Script

```bash
# Test on miniImagenet with visualization
python test.py --dataset miniImagenet --method FSCT_cosine \
    --backbone Conv4 --n_way 5 --k_shot 1 \
    --visualize_features
```

### Example 2: With Comprehensive Analysis

```bash
# Include both visualization and feature analysis
python test.py --dataset CUB --method FSCT_cosine \
    --backbone ResNet18 --n_way 5 --k_shot 5 \
    --visualize_features --feature_analysis 1
```

### Example 3: Custom Visualization

```python
from feature_visualizer import visualize_features_from_results
import numpy as np

# Your feature extraction code
features = extract_my_features()  # shape: (n_samples, n_features)
labels = get_my_labels()          # shape: (n_samples,)

# Visualize with custom settings
result = visualize_features_from_results(
    features=features,
    labels=labels,
    show=True,
    save_dir='./my_experiment/visualizations',
    figsize=(20, 14),  # Larger figure
    title_prefix="My Experiment - Epoch 50"
)
```

## Troubleshooting

### Issue: "No module named 'tkinter'"
**Solution**: The TkAgg backend requires tkinter. Either:
1. Install tkinter: `sudo apt-get install python3-tk` (Linux)
2. Use a different backend: `matplotlib.use('Qt5Agg')`
3. Use non-interactive mode: `matplotlib.use('Agg')` and set `show=False`

### Issue: "Could not extract features for visualization"
**Solution**: Ensure your model has either:
- A `parse_feature()` method that returns support and query features
- A `feature()` method that returns extracted features

### Issue: Visualization is too slow
**Solution**: 
- PCA is fast; use it first to verify the pipeline
- t-SNE is slowest; reduce the number of samples if needed
- UMAP is a good balance between speed and quality

### Issue: Classes are overlapping in the visualization
**Interpretation**: This is normal and indicates:
- Similar classes in the feature space
- The model may have difficulty distinguishing these classes
- Consider using the feature analysis (`--feature_analysis 1`) to get quantitative metrics

## Technical Details

### Supported Models
- FewShotTransformer (FSCT)
- CrossTransformer (CTX)
- OptimalFewShot
- Any model with `parse_feature()` or `feature()` methods

### Dependencies
- matplotlib >= 3.0
- scikit-learn >= 0.24 (includes PCA and t-SNE)
- umap-learn >= 0.5
- numpy
- seaborn (optional, for better styling)

### Performance
- PCA: Very fast, ~1 second for 1000 samples
- t-SNE: Slower, ~10-60 seconds depending on sample count
- UMAP: Moderate, ~5-20 seconds

## Advanced Usage

### Accessing Individual Projections

```python
result = visualize_features_from_results(features, labels, show=False)

# Get specific projections
pca_2d = result['projections']['PCA_2D']
umap_3d = result['projections']['UMAP_3D']

# Create custom plots
import matplotlib.pyplot as plt
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels)
plt.title('Custom PCA Plot')
plt.show()
```

### Modifying Plot Appearance

Edit `feature_visualizer.py` functions `_plot_2d_scatter()` and `_plot_3d_scatter()` to customize:
- Colors (change `plt.cm.viridis` to another colormap)
- Point size (change `s=30`)
- Transparency (change `alpha=0.6`)
- Grid style

## References

- PCA: Pearson, K. (1901). On lines and planes of closest fit to systems of points in space.
- t-SNE: van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE.
- UMAP: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection.
