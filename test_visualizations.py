"""
Test script for feature analysis visualizations.
"""

import numpy as np
import os
from feature_analysis import (
    visualize_embedding_space,
    visualize_attention_maps,
    visualize_weight_distributions,
    visualize_feature_analysis
)

def test_embedding_visualization():
    """Test embedding space visualization"""
    print("\n" + "="*60)
    print("Testing Embedding Space Visualization")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 64
    n_classes = 5
    
    features = np.random.randn(n_samples, n_features)
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Add some structure to make visualization more interesting
    for i in range(n_classes):
        mask = labels == i
        features[mask] += np.random.randn(n_features) * 2
    
    # Test PCA
    fig = visualize_embedding_space(
        features, labels, 
        method='pca', 
        n_components=2,
        save_path='./figures/test_embedding_pca_2d.png',
        title='Test PCA Embedding (2D)'
    )
    
    # Test t-SNE
    fig = visualize_embedding_space(
        features, labels, 
        method='tsne', 
        n_components=2,
        save_path='./figures/test_embedding_tsne_2d.png',
        title='Test t-SNE Embedding (2D)'
    )
    
    # Test 3D
    fig = visualize_embedding_space(
        features, labels, 
        method='pca', 
        n_components=3,
        save_path='./figures/test_embedding_pca_3d.png',
        title='Test PCA Embedding (3D)'
    )
    
    print("✓ Embedding visualization tests passed!")


def test_attention_visualization():
    """Test attention maps visualization"""
    print("\n" + "="*60)
    print("Testing Attention Maps Visualization")
    print("="*60)
    
    # Generate synthetic attention weights
    np.random.seed(42)
    n_queries = 15
    n_support = 25
    
    # Single head attention
    attention_weights = np.random.rand(n_queries, n_support)
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    fig = visualize_attention_maps(
        attention_weights,
        save_path='./figures/test_attention_single.png',
        title='Test Single-Head Attention'
    )
    
    # Multi-head attention
    n_heads = 8
    attention_weights_multi = np.random.rand(n_heads, n_queries, n_support)
    for i in range(n_heads):
        attention_weights_multi[i] = attention_weights_multi[i] / attention_weights_multi[i].sum(axis=1, keepdims=True)
    
    fig = visualize_attention_maps(
        attention_weights_multi,
        save_path='./figures/test_attention_multi.png',
        title='Test Multi-Head Attention'
    )
    
    print("✓ Attention visualization tests passed!")


def test_weight_distributions():
    """Test weight distribution visualization"""
    print("\n" + "="*60)
    print("Testing Weight Distribution Visualization")
    print("="*60)
    
    # Generate synthetic weights
    np.random.seed(42)
    model_weights = {
        'ATTN.input_linear.weight': np.random.randn(512, 64) * 0.1,
        'ATTN.output_linear.weight': np.random.randn(64, 512) * 0.05,
        'FFN.1.weight': np.random.randn(512, 64) * 0.08,
        'FFN.3.weight': np.random.randn(64, 512) * 0.06,
        'linear.1.weight': np.random.randn(64, 64) * 0.12,
    }
    
    fig = visualize_weight_distributions(
        model_weights,
        save_path='./figures/test_weight_distributions.png',
        title='Test Model Weight Distributions'
    )
    
    print("✓ Weight distribution visualization tests passed!")


def test_comprehensive_visualization():
    """Test comprehensive visualization function"""
    print("\n" + "="*60)
    print("Testing Comprehensive Visualization")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 64
    n_classes = 5
    
    features = np.random.randn(n_samples, n_features)
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Add structure
    for i in range(n_classes):
        mask = labels == i
        features[mask] += np.random.randn(n_features) * 2
    
    # Generate attention weights
    n_queries = 15
    n_support = 25
    n_heads = 8
    attention_weights = np.random.rand(n_heads, n_queries, n_support)
    for i in range(n_heads):
        attention_weights[i] = attention_weights[i] / attention_weights[i].sum(axis=1, keepdims=True)
    
    # Generate model weights
    model_weights = {
        'ATTN.weight': np.random.randn(512, 64) * 0.1,
        'FFN.weight': np.random.randn(512, 64) * 0.08,
        'linear.weight': np.random.randn(64, 64) * 0.12,
    }
    
    # Run comprehensive visualization
    figures = visualize_feature_analysis(
        features=features,
        labels=labels,
        attention_weights=attention_weights,
        model_weights=model_weights,
        save_dir='./figures/comprehensive_test',
        methods=['pca', 'tsne']
    )
    
    print(f"✓ Generated {len(figures)} visualizations")
    print("✓ Comprehensive visualization tests passed!")


if __name__ == '__main__':
    # Create output directory
    os.makedirs('./figures', exist_ok=True)
    
    print("\n" + "="*60)
    print("FEATURE ANALYSIS VISUALIZATION TESTS")
    print("="*60)
    
    try:
        test_embedding_visualization()
        test_attention_visualization()
        test_weight_distributions()
        test_comprehensive_visualization()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print(f"\nVisualization outputs saved to ./figures/")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
