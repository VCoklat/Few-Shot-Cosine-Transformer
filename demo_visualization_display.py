"""
Demo script showing how visualization functions now display plots in addition to saving them.

Run this script to see visualizations appear (in interactive environments) and also get saved to files.
"""

import numpy as np
import os
from feature_analysis import (
    visualize_embedding_space,
    visualize_attention_maps,
    visualize_weight_distributions
)

def main():
    print("="*70)
    print("VISUALIZATION DISPLAY DEMO")
    print("="*70)
    print("\nThis demo shows that visualizations are now displayed in addition to")
    print("being saved to files. In an interactive environment (e.g., Jupyter),")
    print("plots will appear automatically.\n")
    
    # Create output directory
    os.makedirs('./figures/demo', exist_ok=True)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 150
    n_features = 64
    n_classes = 5
    
    # Create features with some structure
    features = np.random.randn(n_samples, n_features)
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Add class-specific offsets for better visualization
    for i in range(n_classes):
        mask = labels == i
        features[mask] += np.random.randn(n_features) * 3
    
    # Demo 1: Embedding visualization with display
    print("\n" + "-"*70)
    print("Demo 1: Embedding Space Visualization")
    print("-"*70)
    print("Creating a PCA embedding visualization...")
    print("- Will be DISPLAYED in interactive environments")
    print("- Will be SAVED to ./figures/demo/demo_embedding.png")
    
    fig = visualize_embedding_space(
        features=features,
        labels=labels,
        method='pca',
        n_components=2,
        save_path='./figures/demo/demo_embedding.png',
        title='Demo: Feature Space (PCA)',
        show=True  # This makes it display!
    )
    
    # Demo 2: Attention maps with display
    print("\n" + "-"*70)
    print("Demo 2: Attention Map Visualization")
    print("-"*70)
    print("Creating attention weight heatmap...")
    print("- Will be DISPLAYED in interactive environments")
    print("- Will be SAVED to ./figures/demo/demo_attention.png")
    
    # Generate synthetic attention weights
    n_queries = 15
    n_support = 25
    attention_weights = np.random.rand(n_queries, n_support)
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    fig = visualize_attention_maps(
        attention_weights=attention_weights,
        save_path='./figures/demo/demo_attention.png',
        title='Demo: Attention Weights',
        show=True  # This makes it display!
    )
    
    # Demo 3: Weight distributions with display
    print("\n" + "-"*70)
    print("Demo 3: Weight Distribution Visualization")
    print("-"*70)
    print("Creating weight distribution histograms...")
    print("- Will be DISPLAYED in interactive environments")
    print("- Will be SAVED to ./figures/demo/demo_weights.png")
    
    # Generate synthetic model weights
    model_weights = {
        'layer1.weight': np.random.randn(256, 64) * 0.1,
        'layer2.weight': np.random.randn(128, 256) * 0.08,
        'layer3.weight': np.random.randn(64, 128) * 0.12,
    }
    
    fig = visualize_weight_distributions(
        model_weights=model_weights,
        save_path='./figures/demo/demo_weights.png',
        title='Demo: Model Weight Distributions',
        show=True  # This makes it display!
    )
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\n✓ All visualizations have been:")
    print("  1. DISPLAYED in your environment (if interactive)")
    print("  2. SAVED to ./figures/demo/")
    print("\nYou can control this behavior with the 'show' parameter:")
    print("  - show=True  : Display AND save (default)")
    print("  - show=False : Only save, don't display")
    print("\nThis is useful for:")
    print("  - Jupyter notebooks: Set show=True to see plots inline")
    print("  - Batch processing: Set show=False to avoid opening windows")
    print("  - Interactive scripts: Set show=True to see results immediately")
    print("="*70)

if __name__ == '__main__':
    main()
