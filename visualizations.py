"""
Comprehensive Visualization Module for Few-Shot Learning Models

This module provides advanced visualization functions for analyzing and understanding
few-shot learning models, including:
- Embedding space visualization
- Attention map visualization
- Weight distribution analysis
- Feature analysis visualization
- Integration with FeatureVisualizer class

Author: Few-Shot Cosine Transformer Team
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

# Import existing feature analysis module
from feature_analysis import comprehensive_feature_analysis
from feature_visualizer import FeatureVisualizer


def visualize_embedding_space(
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
) -> Union[go.Figure, plt.Figure]:
    """
    Visualize embedding space using dimensionality reduction techniques.
    
    This function reduces high-dimensional feature embeddings to 2D or 3D space
    for visualization, helping to understand class separability and feature quality.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        labels: Class labels for each sample of shape (n_samples,)
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        n_components: Number of dimensions to reduce to (2 or 3)
        title: Custom title for the plot
        save_path: Path to save the visualization (HTML for interactive, PNG for static)
        interactive: If True, use Plotly for interactive plots; otherwise use matplotlib
        perplexity: Perplexity parameter for t-SNE (default: 30)
        n_neighbors: Number of neighbors for UMAP (default: 15)
        **kwargs: Additional arguments passed to the reduction method
        
    Returns:
        Plotly Figure or Matplotlib Figure object
        
    Example:
        >>> features = model.extract_features(data_loader)
        >>> labels = np.array([0, 0, 1, 1, 2, 2])
        >>> fig = visualize_embedding_space(features, labels, method='tsne')
        >>> fig.show()
    """
    print(f"Visualizing embedding space using {method.upper()}...")
    
    # Validate inputs
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of samples in features ({features.shape[0]}) "
                        f"doesn't match labels ({labels.shape[0]})")
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            **kwargs
        )
        embeddings = reducer.fit_transform(features)
        method_name = "t-SNE"
        
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
        embeddings = reducer.fit_transform(features)
        # Get variance explained
        var_explained = reducer.explained_variance_ratio_
        method_name = f"PCA (Var: {sum(var_explained):.2%})"
        
    elif method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=42,
            **kwargs
        )
        embeddings = reducer.fit_transform(features)
        method_name = "UMAP"
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'tsne', 'pca', or 'umap'")
    
    # Set default title
    if title is None:
        title = f"Embedding Space Visualization - {method_name}"
    
    # Create DataFrame for plotting
    df = pd.DataFrame()
    df['Component 1'] = embeddings[:, 0]
    df['Component 2'] = embeddings[:, 1]
    if n_components == 3:
        df['Component 3'] = embeddings[:, 2]
    df['Class'] = labels.astype(str)
    
    if interactive:
        # Create interactive Plotly visualization
        if n_components == 2:
            fig = px.scatter(
                df, x='Component 1', y='Component 2', color='Class',
                title=title,
                hover_data={'Class': True},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        else:  # 3D
            fig = px.scatter_3d(
                df, x='Component 1', y='Component 2', z='Component 3',
                color='Class',
                title=title,
                hover_data={'Class': True},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        
        # Update layout
        fig.update_layout(
            legend_title="Class",
            template="plotly_white",
            font=dict(size=12),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive plot to {save_path}")
        
        return fig
        
    else:
        # Create static matplotlib visualization
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = sns.scatterplot(
                data=df, x='Component 1', y='Component 2', hue='Class',
                palette='Set1', s=50, alpha=0.7, ax=ax
            )
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
            
        else:  # 3D
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    embeddings[mask, 2],
                    c=[colors[i]], label=str(label), s=50, alpha=0.7
                )
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(title='Class')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved static plot to {save_path}")
        
        return fig


def visualize_attention_maps(
    attention_weights: Union[torch.Tensor, np.ndarray],
    query_labels: Optional[np.ndarray] = None,
    support_labels: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    show_values: bool = False
) -> plt.Figure:
    """
    Visualize attention maps from transformer models.
    
    This function creates heatmaps to visualize attention weights between
    query and support samples, helping understand which support samples
    the model attends to for each query.
    
    Args:
        attention_weights: Attention weights tensor/array of shape 
                          (n_queries, n_supports) or (n_heads, n_queries, n_supports)
        query_labels: Optional labels for query samples
        support_labels: Optional labels for support samples
        title: Custom title for the plot
        save_path: Path to save the visualization
        figsize: Figure size as (width, height)
        cmap: Colormap for the heatmap
        show_values: If True, display attention values on the heatmap
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> # Single attention map
        >>> attn = torch.randn(75, 25).softmax(dim=-1)
        >>> fig = visualize_attention_maps(attn)
        >>> 
        >>> # Multi-head attention
        >>> attn = torch.randn(8, 75, 25).softmax(dim=-1)
        >>> fig = visualize_attention_maps(attn)
    """
    print("Visualizing attention maps...")
    
    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Handle multi-head attention
    if attention_weights.ndim == 3:
        n_heads = attention_weights.shape[0]
        # Create subplots for each head
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]))
        if n_heads == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            attn_map = attention_weights[head_idx]
            
            # Create heatmap
            sns.heatmap(
                attn_map,
                ax=ax,
                cmap=cmap,
                cbar=True,
                square=False,
                annot=show_values,
                fmt='.2f' if show_values else '',
                xticklabels=support_labels if support_labels is not None else False,
                yticklabels=query_labels if query_labels is not None else False
            )
            ax.set_title(f'Head {head_idx + 1}', fontsize=10)
            ax.set_xlabel('Support Samples')
            ax.set_ylabel('Query Samples')
        
        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].axis('off')
        
        if title is None:
            title = f'Attention Maps ({n_heads} heads)'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
    else:
        # Single attention map
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            attention_weights,
            ax=ax,
            cmap=cmap,
            cbar=True,
            square=False,
            annot=show_values,
            fmt='.2f' if show_values else '',
            xticklabels=support_labels if support_labels is not None else False,
            yticklabels=query_labels if query_labels is not None else False
        )
        
        if title is None:
            title = 'Attention Map'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Support Samples', fontsize=12)
        ax.set_ylabel('Query Samples', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention maps to {save_path}")
    
    return fig


def visualize_weight_distributions(
    model: nn.Module,
    layer_types: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    bins: int = 50
) -> plt.Figure:
    """
    Visualize the distribution of weights in a neural network model.
    
    This function creates histograms showing weight distributions across
    different layers, helping identify issues like vanishing/exploding gradients
    or poor initialization.
    
    Args:
        model: PyTorch model
        layer_types: List of layer types to visualize (e.g., ['Conv2d', 'Linear'])
                    If None, visualizes all layers with weights
        title: Custom title for the plot
        save_path: Path to save the visualization
        figsize: Figure size as (width, height)
        bins: Number of bins for histograms
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> model = MyFewShotModel()
        >>> fig = visualize_weight_distributions(model, layer_types=['Linear', 'Conv2d'])
        >>> plt.show()
    """
    print("Extracting and visualizing weight distributions...")
    
    # Extract weights from model
    weight_data = []
    layer_names = []
    
    for name, module in model.named_modules():
        # Check if we should include this layer
        include_layer = False
        if layer_types is None:
            include_layer = hasattr(module, 'weight') and module.weight is not None
        else:
            include_layer = any(layer_type in str(type(module).__name__) 
                              for layer_type in layer_types)
            include_layer = include_layer and hasattr(module, 'weight') and module.weight is not None
        
        if include_layer:
            weights = module.weight.detach().cpu().numpy().flatten()
            weight_data.append(weights)
            layer_names.append(f"{name}\n({type(module).__name__})")
    
    if not weight_data:
        print("No weights found to visualize!")
        return None
    
    # Create subplots
    n_layers = len(weight_data)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot histograms
    for idx, (weights, name) in enumerate(zip(weight_data, layer_names)):
        ax = axes[idx]
        
        # Compute statistics
        mean = np.mean(weights)
        std = np.std(weights)
        
        # Plot histogram
        ax.hist(weights, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        ax.axvline(mean + std, color='orange', linestyle='--', linewidth=1, 
                   label=f'Std: {std:.4f}')
        ax.axvline(mean - std, color='orange', linestyle='--', linewidth=1)
        
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    if title is None:
        title = f'Weight Distributions Across {n_layers} Layers'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved weight distributions to {save_path}")
    
    return fig


def visualize_feature_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create comprehensive visualization of feature space analysis.
    
    This function performs multiple analyses on the feature space and creates
    a comprehensive dashboard including:
    - Feature collapse detection
    - Feature utilization metrics
    - Diversity scores
    - Intra-class consistency
    - Confusing class pairs
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        labels: Class labels for each sample
        title: Custom title for the plot
        save_path: Path to save the visualization
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> features = model.extract_features(data_loader)
        >>> labels = np.array([0, 0, 1, 1, 2, 2])
        >>> fig = visualize_feature_analysis(features, labels)
        >>> plt.show()
    """
    print("Performing comprehensive feature analysis...")
    
    # Perform comprehensive analysis
    analysis_results = comprehensive_feature_analysis(features, labels)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Feature Collapse Visualization (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    collapse_data = analysis_results['feature_collapse']
    std_per_dim = np.array(collapse_data['std_per_dimension'])
    ax1.plot(std_per_dim, linewidth=1)
    ax1.axhline(1e-4, color='red', linestyle='--', label='Collapse Threshold')
    ax1.set_xlabel('Feature Dimension')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_title(f"Feature Collapse\n({collapse_data['collapsed_dimensions']}/{collapse_data['total_dimensions']} collapsed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature Utilization (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    util_data = analysis_results['feature_utilization']
    metrics = ['Mean', 'Std', 'Min', 'Max']
    values = [
        util_data['mean_utilization'],
        util_data['std_utilization'],
        util_data['min_utilization'],
        util_data['max_utilization']
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Utilization Score')
    ax2.set_title('Feature Utilization')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Diversity Score per Class (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    diversity_data = analysis_results['diversity_score']
    if diversity_data['per_class_diversity']:
        per_class_div = diversity_data['per_class_diversity']
        ax3.bar(range(len(per_class_div)), per_class_div, color='steelblue', 
                alpha=0.7, edgecolor='black')
        ax3.axhline(diversity_data['mean_diversity'], color='red', linestyle='--',
                   label='Mean')
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Diversity Score (CV)')
        ax3.set_title('Per-Class Diversity')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Feature Redundancy (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    redun_data = analysis_results['feature_redundancy']
    categories = ['Total\nFeatures', 'Effective\nDims (95%)', 'High Corr\nPairs', 'Moderate\nCorr Pairs']
    values = [
        redun_data['total_features'],
        redun_data['effective_dimensions_95pct'],
        redun_data['high_correlation_pairs'],
        redun_data['moderate_correlation_pairs']
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    ax4.bar(range(len(categories)), values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.set_ylabel('Count')
    ax4.set_title('Feature Redundancy Analysis')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Intra-class Consistency (middle-middle)
    ax5 = fig.add_subplot(gs[1, 1])
    consist_data = analysis_results['intraclass_consistency']
    consistency_types = ['Euclidean', 'Cosine', 'Combined']
    consistency_values = [
        consist_data['mean_euclidean_consistency'],
        consist_data['mean_cosine_consistency'],
        consist_data['mean_combined_consistency']
    ]
    colors = ['#9b59b6', '#1abc9c', '#34495e']
    ax5.bar(consistency_types, consistency_values, color=colors, alpha=0.7, 
            edgecolor='black')
    ax5.set_ylabel('Consistency Score')
    ax5.set_title('Intra-class Consistency')
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Confusing Class Pairs (middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    confuse_data = analysis_results['confusing_pairs']
    if confuse_data['most_confusing_pairs']:
        top_pairs = confuse_data['most_confusing_pairs'][:5]
        pair_labels = [f"{p['class_1']}-{p['class_2']}" for p in top_pairs]
        distances = [p['distance'] for p in top_pairs]
        ax6.barh(range(len(pair_labels)), distances, color='coral', alpha=0.7, 
                 edgecolor='black')
        ax6.set_yticks(range(len(pair_labels)))
        ax6.set_yticklabels(pair_labels)
        ax6.set_xlabel('Inter-centroid Distance')
        ax6.set_title('Most Confusing Class Pairs')
        ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. Class Imbalance (bottom-left)
    ax7 = fig.add_subplot(gs[2, 0])
    imbalance_data = analysis_results['imbalance_ratio']
    if imbalance_data['per_class_counts']:
        classes = list(imbalance_data['per_class_counts'].keys())
        counts = list(imbalance_data['per_class_counts'].values())
        ax7.bar(classes, counts, color='mediumseagreen', alpha=0.7, edgecolor='black')
        ax7.axhline(imbalance_data['mean_class_samples'], color='red', linestyle='--',
                   label='Mean')
        ax7.set_xlabel('Class')
        ax7.set_ylabel('Sample Count')
        ax7.set_title(f"Class Distribution (Imbalance: {imbalance_data['imbalance_ratio']:.3f})")
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Summary Statistics (bottom-middle and bottom-right)
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    # Create summary text
    summary_text = "Feature Analysis Summary\n" + "="*50 + "\n\n"
    summary_text += f"Feature Collapse:\n"
    summary_text += f"  • Collapsed dimensions: {collapse_data['collapsed_dimensions']}/{collapse_data['total_dimensions']}\n"
    summary_text += f"  • Collapse ratio: {collapse_data['collapse_ratio']:.4f}\n\n"
    
    summary_text += f"Feature Utilization:\n"
    summary_text += f"  • Mean utilization: {util_data['mean_utilization']:.4f}\n"
    summary_text += f"  • Low utilization dims: {util_data['low_utilization_dims']}\n\n"
    
    summary_text += f"Feature Redundancy:\n"
    summary_text += f"  • Effective dimensions (95%): {redun_data['effective_dimensions_95pct']}\n"
    summary_text += f"  • Dimensionality reduction: {redun_data['dimensionality_reduction_ratio']:.4f}\n"
    summary_text += f"  • Mean correlation: {redun_data['mean_abs_correlation']:.4f}\n\n"
    
    summary_text += f"Intra-class Consistency:\n"
    summary_text += f"  • Combined consistency: {consist_data['mean_combined_consistency']:.4f}\n\n"
    
    summary_text += f"Inter-class Separation:\n"
    summary_text += f"  • Mean distance: {confuse_data['mean_intercentroid_distance']:.4f}\n"
    summary_text += f"  • Min distance: {confuse_data['min_intercentroid_distance']:.4f}\n\n"
    
    summary_text += f"Class Balance:\n"
    summary_text += f"  • Imbalance ratio: {imbalance_data['imbalance_ratio']:.4f}\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    if title is None:
        title = 'Comprehensive Feature Space Analysis'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature analysis to {save_path}")
    
    return fig


# Integration with FeatureVisualizer class
def enhance_feature_visualizer():
    """
    Enhance the existing FeatureVisualizer class with new visualization methods.
    
    This function adds the new visualization methods to the FeatureVisualizer class,
    making them accessible as instance methods.
    """
    # Add new methods to FeatureVisualizer
    FeatureVisualizer.visualize_embedding_space = lambda self, **kwargs: visualize_embedding_space(
        self.features, self.labels, **kwargs
    )
    
    FeatureVisualizer.visualize_feature_analysis = lambda self, **kwargs: visualize_feature_analysis(
        self.features, self.labels, **kwargs
    )
    
    print("FeatureVisualizer enhanced with new visualization methods!")


# Main demonstration function
def demo_visualizations():
    """
    Demonstrate all visualization functions with synthetic data.
    
    This function creates synthetic data and shows how to use each visualization function.
    Useful for testing and understanding the visualization capabilities.
    """
    print("="*60)
    print("Demonstration of Visualization Functions")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 300
    n_features = 128
    n_classes = 5
    
    # Generate features with some structure
    features = []
    labels = []
    for i in range(n_classes):
        class_center = np.random.randn(n_features) * 2
        class_samples = class_center + np.random.randn(n_samples // n_classes, n_features) * 0.5
        features.append(class_samples)
        labels.extend([i] * (n_samples // n_classes))
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    print(f"\nGenerated synthetic data:")
    print(f"  • Features shape: {features.shape}")
    print(f"  • Labels shape: {labels.shape}")
    print(f"  • Number of classes: {n_classes}")
    
    # 1. Visualize embedding space
    print("\n1. Creating embedding space visualization...")
    fig1 = visualize_embedding_space(
        features, labels, method='tsne', 
        title='Demo: t-SNE Embedding Space',
        interactive=False
    )
    
    # 2. Visualize attention maps (synthetic)
    print("\n2. Creating attention maps visualization...")
    attention = np.random.rand(8, 75, 25)  # 8 heads, 75 queries, 25 supports
    attention = attention / attention.sum(axis=2, keepdims=True)  # Normalize
    fig2 = visualize_attention_maps(
        attention,
        title='Demo: Multi-Head Attention Maps'
    )
    
    # 3. Visualize weight distributions (create a simple model)
    print("\n3. Creating weight distribution visualization...")
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.conv2 = nn.Conv2d(64, 128, 3)
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 10)
    
    model = SimpleModel()
    fig3 = visualize_weight_distributions(
        model,
        layer_types=['Conv2d', 'Linear'],
        title='Demo: Weight Distributions'
    )
    
    # 4. Visualize feature analysis
    print("\n4. Creating feature analysis visualization...")
    fig4 = visualize_feature_analysis(
        features, labels,
        title='Demo: Comprehensive Feature Analysis'
    )
    
    print("\n" + "="*60)
    print("Demo completed! Close the figures to exit.")
    print("="*60)
    
    plt.show()


if __name__ == "__main__":
    # Run demonstration
    demo_visualizations()
