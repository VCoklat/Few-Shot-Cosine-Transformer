# Visualization Module Documentation

This document provides comprehensive documentation for the visualization functions available in the Few-Shot Cosine Transformer repository.

## Overview

The `visualizations.py` module provides advanced visualization capabilities for analyzing and understanding few-shot learning models. It includes functions for:

1. **Embedding Space Visualization** - Visualize high-dimensional feature embeddings
2. **Attention Map Visualization** - Visualize transformer attention weights
3. **Weight Distribution Analysis** - Analyze neural network weight distributions
4. **Feature Analysis Visualization** - Comprehensive feature space analysis dashboard
5. **FeatureVisualizer Integration** - Enhanced FeatureVisualizer class methods

## Installation

Ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- numpy
- matplotlib
- seaborn
- torch
- scikit-learn
- plotly
- pandas
- umap-learn (optional, for UMAP visualization)

## Functions

### 1. `visualize_embedding_space()`

Visualize high-dimensional feature embeddings using dimensionality reduction techniques.

**Signature:**
```python
visualize_embedding_space(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    n_components: int = 2,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    interactive: bool = True,
    perplexity: int = 30,
    n_neighbors: int = 15,
    **kwargs
) -> Union[go.Figure, plt.Figure]
```

**Parameters:**
- `features`: Feature matrix of shape (n_samples, n_features)
- `labels`: Class labels for each sample of shape (n_samples,)
- `method`: Dimensionality reduction method ('tsne', 'pca', 'umap')
- `n_components`: Number of dimensions to reduce to (2 or 3)
- `title`: Custom title for the plot
- `save_path`: Path to save the visualization (HTML for interactive, PNG for static)
- `interactive`: If True, use Plotly for interactive plots; otherwise use matplotlib
- `perplexity`: Perplexity parameter for t-SNE (default: 30)
- `n_neighbors`: Number of neighbors for UMAP (default: 15)
- `**kwargs`: Additional arguments passed to the reduction method

**Returns:**
- Plotly Figure (if interactive=True) or Matplotlib Figure object

**Example Usage:**
```python
from visualizations import visualize_embedding_space
import numpy as np

# Extract features from your model
features = model.extract_features(data_loader)  # Shape: (300, 512)
labels = np.array([0, 0, 1, 1, 2, 2, ...])      # Shape: (300,)

# Create t-SNE visualization
fig = visualize_embedding_space(
    features=features,
    labels=labels,
    method='tsne',
    n_components=2,
    title='5-way Classification: t-SNE Embedding',
    save_path='embedding_tsne.html',
    interactive=True,
    perplexity=30
)
fig.show()  # Display interactive plot

# Create 3D PCA visualization
fig_3d = visualize_embedding_space(
    features=features,
    labels=labels,
    method='pca',
    n_components=3,
    interactive=True
)

# Create UMAP visualization (if available)
fig_umap = visualize_embedding_space(
    features=features,
    labels=labels,
    method='umap',
    n_neighbors=15
)
```

**Use Cases:**
- Visualize class separability in feature space
- Identify clusters and outliers
- Compare different model architectures
- Analyze the effect of training on feature representations

---

### 2. `visualize_attention_maps()`

Visualize attention weights from transformer models to understand query-support relationships.

**Signature:**
```python
visualize_attention_maps(
    attention_weights: Union[torch.Tensor, np.ndarray],
    query_labels: Optional[np.ndarray] = None,
    support_labels: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    show_values: bool = False
) -> plt.Figure
```

**Parameters:**
- `attention_weights`: Attention weights tensor/array of shape (n_queries, n_supports) or (n_heads, n_queries, n_supports)
- `query_labels`: Optional labels for query samples
- `support_labels`: Optional labels for support samples
- `title`: Custom title for the plot
- `save_path`: Path to save the visualization
- `figsize`: Figure size as (width, height)
- `cmap`: Colormap for the heatmap
- `show_values`: If True, display attention values on the heatmap

**Returns:**
- Matplotlib Figure object

**Example Usage:**
```python
from visualizations import visualize_attention_maps
import torch

# Extract attention weights from your model during forward pass
# Single-head attention
attention_single = model.get_attention_weights(query, support)  # Shape: (75, 25)

fig = visualize_attention_maps(
    attention_weights=attention_single,
    title='Query-Support Attention Weights',
    save_path='attention_map.png',
    show_values=False
)

# Multi-head attention
attention_multi = model.get_multihead_attention_weights(query, support)  # Shape: (8, 75, 25)

fig = visualize_attention_maps(
    attention_weights=attention_multi,
    title='Multi-Head Attention (8 heads)',
    save_path='attention_multihead.png'
)

# With labels
query_labels = ['Query_0', 'Query_1', ...]
support_labels = ['Support_0', 'Support_1', ...]

fig = visualize_attention_maps(
    attention_weights=attention_single,
    query_labels=query_labels,
    support_labels=support_labels,
    save_path='attention_labeled.png'
)
```

**Use Cases:**
- Understand which support samples the model attends to for each query
- Compare attention patterns across different heads
- Identify attention patterns for correct vs. incorrect predictions
- Debug transformer-based few-shot learning models

---

### 3. `visualize_weight_distributions()`

Visualize the distribution of weights across different layers of a neural network.

**Signature:**
```python
visualize_weight_distributions(
    model: nn.Module,
    layer_types: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    bins: int = 50
) -> plt.Figure
```

**Parameters:**
- `model`: PyTorch model
- `layer_types`: List of layer types to visualize (e.g., ['Conv2d', 'Linear']). If None, visualizes all layers with weights
- `title`: Custom title for the plot
- `save_path`: Path to save the visualization
- `figsize`: Figure size as (width, height)
- `bins`: Number of bins for histograms

**Returns:**
- Matplotlib Figure object

**Example Usage:**
```python
from visualizations import visualize_weight_distributions
import torch.nn as nn

# Load your trained model
model = load_trained_model()

# Visualize all layers
fig = visualize_weight_distributions(
    model=model,
    title='Weight Distributions: All Layers',
    save_path='weights_all.png'
)

# Visualize only convolutional layers
fig_conv = visualize_weight_distributions(
    model=model,
    layer_types=['Conv2d'],
    title='Convolutional Layer Weights',
    save_path='weights_conv.png'
)

# Visualize only linear/transformer layers
fig_linear = visualize_weight_distributions(
    model=model,
    layer_types=['Linear'],
    title='Linear/Transformer Layer Weights',
    save_path='weights_linear.png'
)
```

**Use Cases:**
- Detect vanishing or exploding gradients
- Verify proper weight initialization
- Compare weight distributions before and after training
- Identify layers that may need regularization

---

### 4. `visualize_feature_analysis()`

Create a comprehensive dashboard of feature space analysis including multiple metrics.

**Signature:**
```python
visualize_feature_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure
```

**Parameters:**
- `features`: Feature matrix of shape (n_samples, n_features)
- `labels`: Class labels for each sample
- `title`: Custom title for the plot
- `save_path`: Path to save the visualization
- `figsize`: Figure size as (width, height)

**Returns:**
- Matplotlib Figure object

**Example Usage:**
```python
from visualizations import visualize_feature_analysis
import numpy as np

# Extract features and labels
features, labels = extract_features_from_model(model, data_loader)

# Create comprehensive analysis dashboard
fig = visualize_feature_analysis(
    features=features,
    labels=labels,
    title='Feature Space Analysis Dashboard',
    save_path='analysis_dashboard.png',
    figsize=(16, 12)
)
```

**Dashboard Components:**
1. **Feature Collapse Detection**: Shows dimensions with collapsed variance
2. **Feature Utilization**: Metrics on how well features are being used
3. **Per-Class Diversity**: Diversity scores for each class
4. **Feature Redundancy**: Analysis of correlated features and effective dimensionality
5. **Intra-Class Consistency**: Consistency of features within classes
6. **Confusing Class Pairs**: Identification of hard-to-separate class pairs
7. **Class Distribution**: Sample distribution across classes
8. **Summary Statistics**: Text summary of all metrics

**Use Cases:**
- Comprehensive model evaluation
- Feature space quality assessment
- Identify potential improvements (e.g., reduce redundancy, improve separability)
- Compare different models or training strategies

---

### 5. `enhance_feature_visualizer()`

Enhance the existing `FeatureVisualizer` class with new visualization methods.

**Signature:**
```python
enhance_feature_visualizer() -> None
```

**Example Usage:**
```python
from feature_visualizer import FeatureVisualizer
from visualizations import enhance_feature_visualizer

# Enhance the FeatureVisualizer class
enhance_feature_visualizer()

# Create visualizer instance
visualizer = FeatureVisualizer(model, device='cuda')

# Extract features
features, labels = visualizer.extract_features(test_loader)

# Use the original visualize method
fig1 = visualizer.visualize(method='tsne', interactive=True)

# Use new methods added by enhancement
fig2 = visualizer.visualize_embedding_space(
    method='umap',
    n_components=3,
    interactive=True,
    save_path='embedding_3d.html'
)

fig3 = visualizer.visualize_feature_analysis(
    save_path='feature_analysis.png'
)
```

**Use Cases:**
- Seamless integration with existing code
- Add new capabilities to FeatureVisualizer instances
- Maintain backward compatibility while adding new features

---

## Complete Workflow Example

Here's a complete example showing how to use the visualization functions in a typical few-shot learning workflow:

```python
import torch
from visualizations import (
    visualize_embedding_space,
    visualize_attention_maps,
    visualize_weight_distributions,
    visualize_feature_analysis
)
from feature_visualizer import FeatureVisualizer

# 1. Load your trained model
model = load_trained_model('checkpoint.pth')
model.eval()

# 2. Create feature visualizer
visualizer = FeatureVisualizer(model, device='cuda')

# 3. Extract features from test set
test_loader = get_test_loader(dataset='miniImagenet', n_way=5, k_shot=5)
features, labels = visualizer.extract_features(test_loader)

# 4. Visualize embedding space
fig_embedding = visualize_embedding_space(
    features=features,
    labels=labels,
    method='tsne',
    n_components=2,
    title='Test Set: t-SNE Visualization',
    save_path='results/embedding_tsne.html',
    interactive=True
)

# 5. Extract and visualize attention maps (during model forward pass)
# Assuming your model has a method to return attention weights
x_query, x_support = get_episode()
with torch.no_grad():
    logits, attention = model.forward_with_attention(x_query, x_support)

fig_attention = visualize_attention_maps(
    attention_weights=attention,
    title='Attention Maps: Query-Support',
    save_path='results/attention_maps.png'
)

# 6. Visualize weight distributions
fig_weights = visualize_weight_distributions(
    model=model,
    layer_types=['Conv2d', 'Linear'],
    title='Model Weight Distributions',
    save_path='results/weight_distributions.png'
)

# 7. Comprehensive feature analysis
fig_analysis = visualize_feature_analysis(
    features=features,
    labels=labels,
    title='Comprehensive Feature Analysis',
    save_path='results/feature_analysis_dashboard.png'
)

print("All visualizations created successfully!")
print("Check the 'results/' directory for output files.")
```

---

## Tips and Best Practices

### For Embedding Space Visualization:
- Use **t-SNE** for small to medium datasets (< 10,000 samples) when you want to preserve local structure
- Use **PCA** when you want to preserve global structure or need faster computation
- Use **UMAP** for large datasets or when you want a balance between local and global structure
- Adjust `perplexity` (t-SNE) or `n_neighbors` (UMAP) based on your dataset size
- Use 3D visualizations to capture more variance, but 2D is often easier to interpret

### For Attention Map Visualization:
- Normalize attention weights before visualization (sum to 1 across support dimension)
- Use `show_values=True` for small attention maps to see exact values
- Compare attention patterns between correct and incorrect predictions
- For multi-head attention, look for complementary patterns across heads

### For Weight Distribution Visualization:
- Run before and after training to see how distributions change
- Look for:
  - Gaussian-like distributions (good)
  - Bimodal distributions (may indicate dead neurons)
  - Very narrow distributions (potential underfitting)
  - Very wide distributions (potential overfitting or poor initialization)

### For Feature Analysis:
- Run on both training and test sets to compare
- High collapse ratio (> 0.3) may indicate:
  - Overfitting
  - Poor architecture design
  - Inadequate training
- Low utilization scores may indicate:
  - Redundant features
  - Opportunity for dimensionality reduction
- Low intra-class consistency or small inter-class separation suggests:
  - Model needs more training
  - Architecture improvements needed
  - Feature extraction issues

---

## Troubleshooting

### Import Errors
If you encounter import errors:
```bash
pip install numpy matplotlib seaborn torch scikit-learn plotly pandas
pip install umap-learn  # Optional
```

### Memory Issues
For large datasets:
- Use `method='pca'` instead of t-SNE or UMAP
- Reduce `n_samples` by sampling a subset
- Use `interactive=False` for static plots (lower memory)

### UMAP Not Available
UMAP is optional. Install with:
```bash
pip install umap-learn
```

### Plotly Figures Not Displaying
For interactive figures in Jupyter notebooks:
```python
import plotly.io as pio
pio.renderers.default = "notebook"
```

---

## References

- **t-SNE**: van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE.
- **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection.
- **Attention Visualization**: Vaswani et al. (2017). Attention is All You Need.

---

## Contributing

If you have suggestions for new visualization functions or improvements, please open an issue or submit a pull request.

---

## License

This module is part of the Few-Shot Cosine Transformer repository and follows the same license.
