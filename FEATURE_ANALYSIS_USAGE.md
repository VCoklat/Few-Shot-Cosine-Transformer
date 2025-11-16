# Feature Analysis Usage Guide

## Quick Start

Run comprehensive evaluation with all 8 feature metrics and visualizations in one command:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --FETI 0 --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 \
    --num_epoch 1 --test_iter 600 --feature_analysis 1
```

## What Gets Computed

When `--feature_analysis 1` is set, the system automatically computes and displays:

### 1. Classification Metrics
- Accuracy, Precision, Recall, F1 scores
- Cohen's Kappa, Matthews Correlation Coefficient
- Top-5 Accuracy
- Confusion Matrix
- Per-class F1 scores

### 2. Statistical Confidence
- 95% confidence intervals from per-episode accuracies
- Z-score approximation
- Episode-wise performance distribution

### 3. Feature Space Analysis (8 Metrics)

#### 📉 Feature Collapse Detection
- Number of collapsed dimensions (std < 1e-4)
- Min/Mean/Max standard deviations
- Collapse ratio percentage

#### 📊 Feature Utilization
- Mean, median, min, max utilization
- Percentile-based range (5th to 95th percentile)
- Comparison to theoretical maximum

#### 🎨 Feature Diversity
- Coefficient of variation from class centroids
- Measures how well classes are separated in feature space

#### 🔄 Feature Redundancy
- High correlation pairs (>0.9)
- Medium correlation pairs (>0.7)
- PCA effective dimensionality at 95% variance
- Dimensionality reduction ratio

### 4. **NEW: Visualizations** 🎨

When feature analysis is enabled, the system automatically generates and saves visualizations to `./figures/feature_analysis/`:

#### 🔮 Embedding Space Visualization
- **2D and 3D visualizations** of the feature embedding space
- **Multiple methods**: PCA, t-SNE (and UMAP if installed)
- Shows how well features cluster by class
- Helps identify class separation and feature quality
- Saved as: `embedding_pca_2d.png`, `embedding_pca_3d.png`, `embedding_tsne_2d.png`, `embedding_tsne_3d.png`

#### 🔍 Attention Map Visualization
- **Heatmaps** showing attention weights between support and query samples
- Supports both **single-head and multi-head** attention
- Helps understand which support samples influence query predictions
- Saved as: `attention_maps.png`

#### 📊 Weight Distribution Visualization
- **Histograms** of model weight distributions across layers
- Shows weight statistics (mean, std) for key layers
- Helps identify potential issues like vanishing/exploding gradients
- Useful for model debugging and optimization
- Saved as: `weight_distributions.png`


#### 🎯 Intra-class Consistency
- Mean Euclidean distance to class centroids
- Mean cosine similarity within classes
- Standard deviations for both metrics

#### 🤔 Confusing Class Pairs
- Top-k most confusing pairs based on centroid proximity
- Inter-centroid distance statistics

#### ⚖️ Class Imbalance
- Imbalance ratio (minority/majority)
- Sample count statistics per class

#### 📈 Statistical Confidence
- 95% CI from episode accuracies
- Per-class F1 breakdown
- Confusion matrix

## Example Output

```
============================================================
COMPREHENSIVE FEATURE SPACE ANALYSIS
============================================================

[1/8] Detecting feature collapse...
[2/8] Computing feature utilization...
[3/8] Analyzing feature diversity...
[4/8] Assessing feature redundancy...
[5/8] Evaluating intra-class consistency...
[6/8] Identifying confusing class pairs...
[7/8] Computing class imbalance...
[8/8] Calculating statistical confidence...

✓ Analysis complete!

============================================================
FEATURE ANALYSIS SUMMARY
============================================================

📉 Feature Collapse:
  Collapsed dimensions: 0/64 (0.0%)
  Min/Mean/Max std: 0.860942 / 0.992082 / 1.111071

📊 Feature Utilization:
  Mean: 0.6467
  Median: 0.6512
  Range: [0.5013, 0.8218]

🎨 Feature Diversity:
  Coefficient of Variation: 7.0193
  Num classes: 5

🔄 Feature Redundancy:
  High correlation pairs (>0.9): 0
  Medium correlation pairs (>0.7): 0
  Effective dims (95% variance): 47/64

🎯 Intra-class Consistency:
  Mean Euclidean distance: 7.7568 ± 0.6463
  Mean Cosine similarity: 0.2205 ± 0.1212

🤔 Most Confusing Class Pairs:
  1. Classes 2 ↔ 4: distance = 2.3315
  2. Classes 0 ↔ 1: distance = 2.3827
  3. Classes 1 ↔ 4: distance = 2.3976

⚖️ Class Imbalance:
  Imbalance ratio: 1.0000
  Samples per class: 20 - 20 (mean: 20.0)

📈 Statistical Confidence:
  Mean accuracy: 82.13%
  95% CI: [78.66%, 85.60%]
  Macro F1: 0.1374
```

## Flags

- `--feature_analysis 1` (default): Enable comprehensive feature analysis
- `--feature_analysis 0`: Disable feature analysis (faster evaluation)
- `--comprehensive_eval 1` (default): Use comprehensive evaluation
- `--comprehensive_eval 0`: Use minimal evaluation

## Ablation Studies

See [ABLATION_STUDIES.md](ABLATION_STUDIES.md) for detailed ablation study configurations.

## Requirements

- numpy
- scipy
- scikit-learn
- torch
- psutil
- GPUtil

All required packages are in `requirements.txt`.

## Using Visualization Functions Programmatically

You can also use the visualization functions directly in your code:

### Example 1: Visualize Embedding Space

```python
from feature_analysis import visualize_embedding_space
import numpy as np

# Your feature data
features = np.random.randn(200, 64)  # (n_samples, n_features)
labels = np.repeat(np.arange(5), 40)  # (n_samples,)

# Create PCA visualization
fig = visualize_embedding_space(
    features, labels,
    method='pca',
    n_components=2,
    save_path='./my_embedding_pca.png',
    title='My Feature Embedding'
)

# Create t-SNE visualization
fig = visualize_embedding_space(
    features, labels,
    method='tsne',
    n_components=2,
    perplexity=30,  # Optional parameter
    save_path='./my_embedding_tsne.png'
)
```

### Example 2: Visualize Attention Maps

```python
from feature_analysis import visualize_attention_maps
import numpy as np

# Single-head attention
attention = np.random.rand(15, 25)  # (n_queries, n_support)
attention = attention / attention.sum(axis=1, keepdims=True)

fig = visualize_attention_maps(
    attention,
    save_path='./my_attention.png',
    title='Query-Support Attention'
)

# Multi-head attention
attention_multi = np.random.rand(8, 15, 25)  # (n_heads, n_queries, n_support)
fig = visualize_attention_maps(
    attention_multi,
    save_path='./my_attention_multi.png'
)
```

### Example 3: Visualize Weight Distributions

```python
from feature_analysis import visualize_weight_distributions
import torch

# Extract weights from your model
model_weights = {}
for name, param in model.named_parameters():
    if 'weight' in name:
        model_weights[name] = param.detach().cpu().numpy()

fig = visualize_weight_distributions(
    model_weights,
    save_path='./my_weights.png',
    title='Model Weight Distribution'
)
```

### Example 4: Generate All Visualizations at Once

```python
from feature_analysis import visualize_feature_analysis

figures = visualize_feature_analysis(
    features=features,
    labels=labels,
    attention_weights=attention_weights,  # Optional
    model_weights=model_weights,          # Optional
    save_dir='./my_visualizations',
    methods=['pca', 'tsne', 'umap']       # Choose methods
)

# Returns a dictionary of figure objects
print(f"Generated {len(figures)} visualizations")
```

## Visualization Output Locations

All visualizations generated during feature analysis are saved to:
- **Automatic evaluation**: `./figures/feature_analysis/`
- **Manual usage**: Specify your own `save_path` or `save_dir`

## Dependencies

The visualization features require the following packages (automatically included in `requirements.txt`):
- `matplotlib` - Core plotting library
- `seaborn` - Enhanced visualizations and heatmaps
- `scikit-learn` - PCA and t-SNE dimensionality reduction
- `scipy` - Scientific computing utilities
- `umap-learn` - UMAP dimensionality reduction (optional)

Install with:
```bash
pip install -r requirements.txt
```

## Tips for Best Visualizations

1. **Embedding Space**:
   - Use PCA for quick overview and variance analysis
   - Use t-SNE for better cluster visualization (takes longer)
   - Use UMAP for large datasets (fastest for big data)
   - 2D visualizations are easier to interpret than 3D

2. **Attention Maps**:
   - Look for sparse vs. dense attention patterns
   - Check if attention focuses on relevant support samples
   - Multi-head attention shows different attention strategies

3. **Weight Distributions**:
   - Healthy distributions are roughly Gaussian and centered near zero
   - Watch for very small or very large standard deviations
   - Bimodal distributions may indicate learned structure
