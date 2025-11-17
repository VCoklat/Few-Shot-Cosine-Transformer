"""
Example script demonstrating feature visualization with PCA, t-SNE, and UMAP.

This script shows how to use the visualize_features_from_results function
to generate 2D and 3D projections of feature vectors.

Usage:
    python example_visualization.py
    
Note:
    This example uses the 'Agg' backend which saves plots to files.
    To display plots interactively, change matplotlib.use('Agg') to
    matplotlib.use('TkAgg') or remove the line to use the default backend.
"""

import numpy as np
import matplotlib
# Use 'Agg' for non-interactive (save only) - good for headless environments
# Or use the default backend for interactive display
matplotlib.use('Agg')  
from feature_visualizer import visualize_features_from_results


def generate_synthetic_data(n_classes=5, samples_per_class=20, n_features=50):
    """
    Generate synthetic feature data for visualization demo.
    
    Args:
        n_classes: Number of classes
        samples_per_class: Number of samples per class
        n_features: Dimension of feature vectors
    
    Returns:
        features: numpy array of shape (n_classes * samples_per_class, n_features)
        labels: numpy array of shape (n_classes * samples_per_class,)
    """
    features = []
    labels = []
    
    np.random.seed(42)
    
    for class_id in range(n_classes):
        # Generate class-specific mean
        class_mean = np.random.randn(n_features) * 3
        
        # Generate samples for this class with some noise
        class_features = class_mean + np.random.randn(samples_per_class, n_features)
        
        features.append(class_features)
        labels.extend([class_id] * samples_per_class)
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    return features, labels


if __name__ == '__main__':
    print("="*80)
    print("Feature Visualization Example")
    print("="*80)
    
    # Generate synthetic data
    print("\nGenerating synthetic feature data...")
    features, labels = generate_synthetic_data(
        n_classes=5, 
        samples_per_class=30, 
        n_features=128
    )
    
    print(f"Generated {features.shape[0]} samples with {features.shape[1]} features")
    print(f"Classes: {np.unique(labels)}")
    
    # Visualize features with all projection methods
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    
    result = visualize_features_from_results(
        features=features,
        labels=labels,
        show=False,  # Set to True to display plots using plt.show() if using interactive backend
        save_dir='./figures/example_visualizations',  # Save to this directory
        title_prefix="Synthetic Data"
    )
    
    # Access individual projections if needed
    print("\n" + "="*80)
    print("Projection Statistics")
    print("="*80)
    
    for proj_name, embedding in result['projections'].items():
        print(f"\n{proj_name}:")
        print(f"  Shape: {embedding.shape}")
        print(f"  Mean: {embedding.mean():.4f}")
        print(f"  Std: {embedding.std():.4f}")
    
    print("\n" + "="*80)
    print("✓ Visualization complete!")
    print("="*80)
