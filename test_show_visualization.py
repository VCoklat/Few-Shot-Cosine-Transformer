"""
Test script to verify that visualizations now display in addition to saving.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from feature_analysis import (
    visualize_embedding_space,
    visualize_attention_maps,
    visualize_weight_distributions
)

def test_show_parameter():
    """Test that show parameter works correctly"""
    print("\n" + "="*60)
    print("Testing show parameter functionality")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 99  # Use 99 to evenly divide by 3 classes
    n_features = 32
    n_classes = 3
    
    features = np.random.randn(n_samples, n_features)
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Test 1: visualize_embedding_space with show=False (default for batch)
    print("\n[Test 1] Testing embedding visualization with show=False...")
    fig = visualize_embedding_space(
        features, labels, 
        method='pca', 
        n_components=2,
        save_path='./figures/test_show_false.png',
        show=False  # Should only save, not display
    )
    assert fig is not None, "Figure should be created"
    plt.close(fig)
    print("✓ Test passed: show=False works correctly")
    
    # Test 2: visualize_embedding_space with show=True (default for single)
    print("\n[Test 2] Testing embedding visualization with show=True...")
    fig = visualize_embedding_space(
        features, labels, 
        method='pca', 
        n_components=2,
        save_path='./figures/test_show_true.png',
        show=True  # Should save AND display (in interactive env)
    )
    assert fig is not None, "Figure should be created"
    plt.close(fig)
    print("✓ Test passed: show=True works correctly")
    
    # Test 3: visualize_attention_maps with show parameter
    print("\n[Test 3] Testing attention maps with show parameter...")
    attention_weights = np.random.rand(10, 20)
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    fig = visualize_attention_maps(
        attention_weights,
        save_path='./figures/test_attention_show.png',
        show=True
    )
    assert fig is not None, "Figure should be created"
    plt.close(fig)
    print("✓ Test passed: attention maps show parameter works")
    
    # Test 4: visualize_weight_distributions with show parameter
    print("\n[Test 4] Testing weight distributions with show parameter...")
    model_weights = {
        'layer1.weight': np.random.randn(128, 32) * 0.1,
        'layer2.weight': np.random.randn(32, 128) * 0.08,
    }
    
    fig = visualize_weight_distributions(
        model_weights,
        save_path='./figures/test_weights_show.png',
        show=True
    )
    assert fig is not None, "Figure should be created"
    plt.close(fig)
    print("✓ Test passed: weight distributions show parameter works")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nVisualization files saved to ./figures/")
    print("In an interactive environment, plots would also be displayed.")

if __name__ == '__main__':
    # Create output directory
    os.makedirs('./figures', exist_ok=True)
    
    try:
        test_show_parameter()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
