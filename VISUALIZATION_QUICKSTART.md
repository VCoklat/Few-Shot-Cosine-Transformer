# Visualization Functions - Quick Reference

This section documents the new visualization functions available in `visualizations.py`.

## Quick Start

```python
from visualizations import (
    visualize_embedding_space,
    visualize_attention_maps,
    visualize_weight_distributions,
    visualize_feature_analysis
)
```

## Functions Overview

### 1. visualize_embedding_space()
Visualize high-dimensional features in 2D/3D using dimensionality reduction.

```python
# Example: Visualize test set embeddings
features = model.extract_features(test_loader)
labels = np.array([...])

fig = visualize_embedding_space(
    features, labels, 
    method='tsne',  # or 'pca', 'umap'
    n_components=2,  # or 3 for 3D
    interactive=True,
    save_path='embedding.html'
)
```

### 2. visualize_attention_maps()
Visualize attention weights from transformer models.

```python
# Example: Visualize multi-head attention
attention = model.get_attention_weights(query, support)

fig = visualize_attention_maps(
    attention,
    title='Attention Maps',
    save_path='attention.png'
)
```

### 3. visualize_weight_distributions()
Analyze weight distributions across model layers.

```python
# Example: Visualize all layer weights
fig = visualize_weight_distributions(
    model,
    layer_types=['Conv2d', 'Linear'],
    save_path='weights.png'
)
```

### 4. visualize_feature_analysis()
Create comprehensive feature space analysis dashboard.

```python
# Example: Analyze feature quality
fig = visualize_feature_analysis(
    features, labels,
    save_path='analysis.png'
)
```

## Integration with FeatureVisualizer

```python
from feature_visualizer import FeatureVisualizer
from visualizations import enhance_feature_visualizer

# Enhance existing class
enhance_feature_visualizer()

# Use enhanced methods
visualizer = FeatureVisualizer(model, device='cuda')
features, labels = visualizer.extract_features(test_loader)
fig = visualizer.visualize_embedding_space(method='umap')
```

## Documentation

- **Full API Reference:** See [VISUALIZATION_DOCS.md](VISUALIZATION_DOCS.md)
- **Usage Examples:** See [example_visualizations.py](example_visualizations.py)
- **Implementation Summary:** See [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md)

## Testing

```bash
# Run unit tests
python test_visualizations.py

# Run examples
python example_visualizations.py

# Run demo
python -c "from visualizations import demo_visualizations; demo_visualizations()"
```

---

**Note:** All functions are production-ready and integrate seamlessly with the existing codebase. No new dependencies are required (all packages are already in `requirements.txt`).
