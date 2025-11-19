"""
Example Usage of Visualization Functions

This script demonstrates how to use the visualization functions provided
in visualizations.py for analyzing few-shot learning models.

Author: Few-Shot Cosine Transformer Team
"""

import numpy as np
import torch
import torch.nn as nn

# Import visualization functions
from visualizations import (
    visualize_embedding_space,
    visualize_attention_maps,
    visualize_weight_distributions,
    visualize_feature_analysis,
    enhance_feature_visualizer
)

from feature_visualizer import FeatureVisualizer


def example_1_embedding_space():
    """
    Example 1: Visualizing Embedding Space
    
    This example shows how to visualize high-dimensional feature embeddings
    using different dimensionality reduction techniques.
    """
    print("\n" + "="*60)
    print("Example 1: Embedding Space Visualization")
    print("="*60)
    
    # Simulate extracted features from a model
    # In practice, you would extract these from your trained model
    np.random.seed(42)
    n_samples_per_class = 60
    n_classes = 5
    n_features = 512  # e.g., ResNet feature dimension
    
    features = []
    labels = []
    
    for class_id in range(n_classes):
        # Create features with some class structure
        class_center = np.random.randn(n_features) * 3
        class_features = class_center + np.random.randn(n_samples_per_class, n_features) * 0.8
        features.append(class_features)
        labels.extend([class_id] * n_samples_per_class)
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Visualize using t-SNE (2D)
    print("\nCreating t-SNE visualization...")
    fig1 = visualize_embedding_space(
        features=features,
        labels=labels,
        method='tsne',
        n_components=2,
        title='5-way Classification: t-SNE Embedding',
        save_path='embedding_tsne_2d.html',
        interactive=True,
        perplexity=30
    )
    print("✓ t-SNE visualization created")
    
    # Visualize using PCA (3D)
    print("\nCreating 3D PCA visualization...")
    fig2 = visualize_embedding_space(
        features=features,
        labels=labels,
        method='pca',
        n_components=3,
        title='5-way Classification: PCA Embedding (3D)',
        save_path='embedding_pca_3d.html',
        interactive=True
    )
    print("✓ 3D PCA visualization created")
    
    # Visualize using UMAP (if available)
    try:
        print("\nCreating UMAP visualization...")
        fig3 = visualize_embedding_space(
            features=features,
            labels=labels,
            method='umap',
            n_components=2,
            title='5-way Classification: UMAP Embedding',
            save_path='embedding_umap_2d.html',
            interactive=True,
            n_neighbors=15
        )
        print("✓ UMAP visualization created")
    except ImportError:
        print("⚠ UMAP not available (install with: pip install umap-learn)")


def example_2_attention_maps():
    """
    Example 2: Visualizing Attention Maps
    
    This example shows how to visualize attention weights from transformer models.
    """
    print("\n" + "="*60)
    print("Example 2: Attention Maps Visualization")
    print("="*60)
    
    # Simulate attention weights from a transformer model
    # In practice, you would extract these from your model during forward pass
    
    # Single-head attention example
    print("\nExample 2a: Single-head attention")
    n_queries = 75  # 5-way, 15 query samples per class
    n_supports = 25  # 5-way, 5 support samples per class
    
    attention_single = torch.randn(n_queries, n_supports)
    attention_single = torch.softmax(attention_single, dim=-1)  # Normalize
    
    fig1 = visualize_attention_maps(
        attention_weights=attention_single,
        title='Single-Head Attention: Query-Support Weights',
        save_path='attention_single_head.png',
        figsize=(10, 8),
        show_values=False
    )
    print("✓ Single-head attention visualization created")
    
    # Multi-head attention example
    print("\nExample 2b: Multi-head attention")
    n_heads = 8
    attention_multi = torch.randn(n_heads, n_queries, n_supports)
    attention_multi = torch.softmax(attention_multi, dim=-1)  # Normalize per head
    
    fig2 = visualize_attention_maps(
        attention_weights=attention_multi,
        title='Multi-Head Attention Maps (8 heads)',
        save_path='attention_multi_head.png',
        figsize=(16, 12),
        show_values=False
    )
    print("✓ Multi-head attention visualization created")
    
    # With labels
    print("\nExample 2c: Attention with class labels")
    query_labels = [f"Q{i//15}" for i in range(n_queries)]  # Simplified labels
    support_labels = [f"S{i//5}" for i in range(n_supports)]
    
    fig3 = visualize_attention_maps(
        attention_weights=attention_single,
        query_labels=query_labels,
        support_labels=support_labels,
        title='Attention with Labels',
        save_path='attention_with_labels.png',
        figsize=(12, 10)
    )
    print("✓ Attention with labels visualization created")


def example_3_weight_distributions():
    """
    Example 3: Visualizing Weight Distributions
    
    This example shows how to visualize the distribution of weights
    across different layers of a neural network.
    """
    print("\n" + "="*60)
    print("Example 3: Weight Distributions Visualization")
    print("="*60)
    
    # Create a sample model (similar to few-shot learning architectures)
    class SampleFewShotModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Backbone (simplified Conv4)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            
            # Transformer components
            self.query_proj = nn.Linear(128, 128)
            self.key_proj = nn.Linear(128, 128)
            self.value_proj = nn.Linear(128, 128)
            
            # Classifier
            self.fc = nn.Linear(128, 5)  # 5-way classification
    
    model = SampleFewShotModel()
    
    print("Analyzing weight distributions...")
    
    # Visualize all layers with weights
    print("\nExample 3a: All layers")
    fig1 = visualize_weight_distributions(
        model=model,
        title='Weight Distributions: All Layers',
        save_path='weights_all_layers.png',
        figsize=(15, 10),
        bins=50
    )
    print("✓ All layers visualization created")
    
    # Visualize only convolutional layers
    print("\nExample 3b: Convolutional layers only")
    fig2 = visualize_weight_distributions(
        model=model,
        layer_types=['Conv2d'],
        title='Weight Distributions: Convolutional Layers',
        save_path='weights_conv_layers.png',
        figsize=(12, 8),
        bins=50
    )
    print("✓ Convolutional layers visualization created")
    
    # Visualize only linear layers
    print("\nExample 3c: Linear layers only")
    fig3 = visualize_weight_distributions(
        model=model,
        layer_types=['Linear'],
        title='Weight Distributions: Linear/Transformer Layers',
        save_path='weights_linear_layers.png',
        figsize=(12, 6),
        bins=50
    )
    print("✓ Linear layers visualization created")


def example_4_feature_analysis():
    """
    Example 4: Comprehensive Feature Analysis
    
    This example shows how to perform and visualize comprehensive
    feature space analysis including collapse detection, utilization,
    diversity, consistency, and more.
    """
    print("\n" + "="*60)
    print("Example 4: Comprehensive Feature Analysis")
    print("="*60)
    
    # Simulate extracted features
    np.random.seed(42)
    n_samples_per_class = 50
    n_classes = 5
    n_features = 256
    
    features = []
    labels = []
    
    for class_id in range(n_classes):
        # Create features with varying characteristics
        class_center = np.random.randn(n_features) * 2
        
        # Add some collapsed dimensions (low variance)
        class_center[200:] = 0.001  # Last 56 dimensions have very low variance
        
        class_features = class_center + np.random.randn(n_samples_per_class, n_features) * 0.5
        class_features[:, 200:] += np.random.randn(n_samples_per_class, 56) * 0.0001
        
        features.append(class_features)
        labels.extend([class_id] * n_samples_per_class)
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    print("\nPerforming comprehensive feature analysis...")
    fig = visualize_feature_analysis(
        features=features,
        labels=labels,
        title='Comprehensive Feature Space Analysis Dashboard',
        save_path='feature_analysis_dashboard.png',
        figsize=(16, 12)
    )
    print("✓ Feature analysis dashboard created")
    
    print("\nThe dashboard includes:")
    print("  • Feature collapse detection")
    print("  • Feature utilization metrics")
    print("  • Per-class diversity scores")
    print("  • Feature redundancy analysis")
    print("  • Intra-class consistency")
    print("  • Most confusing class pairs")
    print("  • Class distribution and imbalance")
    print("  • Summary statistics")


def example_5_feature_visualizer_integration():
    """
    Example 5: Integration with FeatureVisualizer Class
    
    This example shows how to use the enhanced FeatureVisualizer class
    with the new visualization methods.
    """
    print("\n" + "="*60)
    print("Example 5: FeatureVisualizer Integration")
    print("="*60)
    
    # Note: This requires a trained model and data loader in practice
    # Here we show the API usage
    
    print("\nEnhancing FeatureVisualizer with new methods...")
    enhance_feature_visualizer()
    print("✓ FeatureVisualizer enhanced")
    
    print("\nUsage example:")
    print("""
    # Assuming you have a trained model and test loader:
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
        interactive=True
    )
    
    fig3 = visualizer.visualize_feature_analysis(
        save_path='analysis.png'
    )
    """)


def main():
    """
    Main function to run all examples.
    
    Note: Some examples create files, so make sure you have write permissions
    in the current directory.
    """
    print("="*60)
    print("Visualization Functions - Example Usage")
    print("="*60)
    print("\nThis script demonstrates how to use all visualization functions")
    print("provided in visualizations.py for few-shot learning analysis.")
    print("\nNote: Install required packages first:")
    print("  pip install numpy matplotlib seaborn torch scikit-learn plotly pandas")
    print("  pip install umap-learn  # Optional, for UMAP visualization")
    
    try:
        # Run examples
        example_1_embedding_space()
        example_2_attention_maps()
        example_3_weight_distributions()
        example_4_feature_analysis()
        example_5_feature_visualizer_integration()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  • embedding_tsne_2d.html")
        print("  • embedding_pca_3d.html")
        print("  • embedding_umap_2d.html (if UMAP available)")
        print("  • attention_single_head.png")
        print("  • attention_multi_head.png")
        print("  • attention_with_labels.png")
        print("  • weights_all_layers.png")
        print("  • weights_conv_layers.png")
        print("  • weights_linear_layers.png")
        print("  • feature_analysis_dashboard.png")
        
    except Exception as e:
        print(f"\n⚠ Error running examples: {e}")
        print("\nMake sure all required packages are installed:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
