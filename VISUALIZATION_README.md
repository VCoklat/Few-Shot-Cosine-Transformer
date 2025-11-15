# Feature Analysis Visualizations

This module provides comprehensive visualization capabilities for analyzing few-shot learning models.

## Overview

Three types of visualizations are automatically generated when feature analysis is enabled:

1. **Embedding Space Visualization** - 2D/3D plots showing feature clustering
2. **Attention Map Visualization** - Heatmaps of attention weights
3. **Weight Distribution Visualization** - Histograms of model parameters

**NEW:** All visualization functions now support a `show` parameter (default: `True`) that displays plots in addition to saving them. This allows visualizations to appear in interactive environments like Jupyter notebooks or when running scripts directly.

## Quick Start

### Enable During Training/Evaluation

Simply add the `--feature_analysis 1` flag:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --feature_analysis 1
```

Visualizations will be automatically saved to `./figures/feature_analysis/`

### Programmatic Usage

```python
from feature_analysis import visualize_embedding_space

# Your data
features = model.extract_features(data)  # shape: (n_samples, n_features)
labels = data.get_labels()  # shape: (n_samples,)

# Display AND save visualization (default behavior)
fig = visualize_embedding_space(
    features=features,
    labels=labels,
    method='tsne',
    save_path='./embedding.png',
    show=True  # Shows plot in interactive environment
)

# Only save, don't display (useful for batch processing)
fig = visualize_embedding_space(
    features=features,
    labels=labels,
    method='pca',
    save_path='./embedding_pca.png',
    show=False  # Only saves to file
)
```

### Batch Visualization

```python
from feature_analysis import visualize_feature_analysis

# Your data
features = model.extract_features(data)
labels = data.get_labels()

# Optional: extract attention and model weights
attention_weights = model.get_attention_weights()  # shape: (n_heads, n_queries, n_support)
model_weights = {name: param.numpy() for name, param in model.named_parameters()}

# Generate all visualizations (show=False by default for batch operations)
figures = visualize_feature_analysis(
    features=features,
    labels=labels,
    attention_weights=attention_weights,
    model_weights=model_weights,
    save_dir='./my_visualizations',
    methods=['pca', 'tsne'],
    show=False  # Set to True to display all plots
)
```

## Visualization Types

### 1. Embedding Space (`visualize_embedding_space`)

Visualizes high-dimensional feature embeddings in 2D or 3D space.

**Methods supported:**
- **PCA** - Fast, shows variance-based projections
- **t-SNE** - Better for cluster visualization, slower
- **UMAP** - Fast for large datasets (requires `umap-learn`)

**Output:**
- `embedding_pca_2d.png` - 2D PCA projection
- `embedding_pca_3d.png` - 3D PCA projection  
- `embedding_tsne_2d.png` - 2D t-SNE projection
- `embedding_tsne_3d.png` - 3D t-SNE projection

**What to look for:**
- Well-separated clusters indicate good feature learning
- Overlapping clusters may indicate confusable classes
- Outliers may indicate anomalous samples

### 2. Attention Maps (`visualize_attention_maps`)

Visualizes attention weights as heatmaps showing which support samples influence query predictions.

**Supports:**
- Single-head attention (one heatmap)
- Multi-head attention (grid of heatmaps)

**Output:**
- `attention_maps.png` - Heatmap(s) of attention weights

**What to look for:**
- Sparse attention = model focuses on specific support samples
- Uniform attention = model averages over all support samples
- Different heads should show different attention patterns

### 3. Weight Distributions (`visualize_weight_distributions`)

Shows histogram distributions of model weights across layers.

**Output:**
- `weight_distributions.png` - Grid of histograms for each layer

**What to look for:**
- Healthy distributions are roughly Gaussian, centered near zero
- Very small std (< 0.01) may indicate dead neurons
- Very large std (> 1.0) may indicate exploding gradients
- Bimodal distributions may indicate learned structure

## API Reference

### `visualize_embedding_space(features, labels, method='pca', n_components=2, save_path=None, show=True, **kwargs)`

Create embedding space visualization.

**Parameters:**
- `features` (np.ndarray): Feature vectors, shape (n_samples, n_features)
- `labels` (np.ndarray): Class labels, shape (n_samples,)
- `method` (str): 'pca', 'tsne', or 'umap'
- `n_components` (int): 2 or 3 for 2D/3D visualization
- `save_path` (str): Path to save figure
- `show` (bool): Whether to display the plot (default: True)
- `**kwargs`: Additional args for the reduction method (e.g., `perplexity` for t-SNE)

**Returns:**
- `matplotlib.figure.Figure` or `None`

### `visualize_attention_maps(attention_weights, save_path=None, query_labels=None, support_labels=None, show=True)`

Create attention weight heatmaps.

**Parameters:**
- `attention_weights` (np.ndarray): Shape (n_queries, n_support) or (n_heads, n_queries, n_support)
- `save_path` (str): Path to save figure
- `query_labels` (np.ndarray): Optional labels for queries
- `support_labels` (np.ndarray): Optional labels for support samples
- `show` (bool): Whether to display the plot (default: True)

**Returns:**
- `matplotlib.figure.Figure` or `None`

### `visualize_weight_distributions(model_weights, save_path=None, layer_names=None, show=True)`

Create weight distribution histograms.

**Parameters:**
- `model_weights` (dict): Mapping of layer names to weight arrays
- `save_path` (str): Path to save figure
- `layer_names` (list): Optional list of specific layers to visualize
- `show` (bool): Whether to display the plot (default: True)

**Returns:**
- `matplotlib.figure.Figure` or `None`

### `visualize_feature_analysis(features, labels, attention_weights=None, model_weights=None, save_dir='./figures', methods=['pca', 'tsne'], show=False)`

Generate all visualizations at once.

**Parameters:**
- `features` (np.ndarray): Feature vectors
- `labels` (np.ndarray): Class labels
- `attention_weights` (np.ndarray, optional): Attention weights
- `model_weights` (dict, optional): Model weights
- `save_dir` (str): Directory to save visualizations
- `methods` (list): Dimensionality reduction methods to use
- `show` (bool): Whether to display plots (default: False for batch operations)

**Returns:**
- `dict`: Mapping of visualization names to Figure objects

## Requirements

```
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
numpy>=1.21.0
umap-learn>=0.5.0  # Optional, for UMAP
```

Install with:
```bash
pip install -r requirements.txt
```

## Examples

See `test_visualizations.py`, `test_show_visualization.py`, and `test_integration.py` for complete examples.

## Tips

1. **For quick analysis**: Use PCA (fastest)
2. **For publication figures**: Use t-SNE (best clustering)
3. **For large datasets**: Use UMAP (best speed/quality tradeoff)
4. **For interpretability**: Use 2D over 3D visualizations
5. **Save high-res**: Add `dpi=300` when saving manually
6. **Interactive environments**: Set `show=True` to display plots in Jupyter notebooks or interactive Python sessions
7. **Batch processing**: Set `show=False` to only save files without displaying (faster for automation)

## Troubleshooting

**"matplotlib not available"**
```bash
pip install matplotlib seaborn
```

**"sklearn not available"**
```bash
pip install scikit-learn scipy
```

**"UMAP not available"**
```bash
pip install umap-learn
```

**Visualizations not generated during training**
- Ensure `--feature_analysis 1` flag is set
- Check that output directory is writable
- Verify dependencies are installed

## Output Directory Structure

```
figures/
└── feature_analysis/
    ├── embedding_pca_2d.png
    ├── embedding_pca_3d.png
    ├── embedding_tsne_2d.png
    ├── embedding_tsne_3d.png
    ├── attention_maps.png
    └── weight_distributions.png
```

## License

Same as parent project.
