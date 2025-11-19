"""
Unit Tests for Visualization Functions

This test file validates the basic functionality of all visualization functions
without requiring a fully trained model or large datasets.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path to import visualizations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    try:
        from visualizations import (
            visualize_embedding_space,
            visualize_attention_maps,
            visualize_weight_distributions,
            visualize_feature_analysis,
            enhance_feature_visualizer
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_visualize_embedding_space():
    """Test embedding space visualization."""
    print("\nTesting visualize_embedding_space()...")
    try:
        from visualizations import visualize_embedding_space
        
        # Create synthetic data
        np.random.seed(42)
        features = np.random.randn(100, 50)
        labels = np.repeat(np.arange(5), 20)
        
        # Test t-SNE 2D
        fig = visualize_embedding_space(
            features, labels, method='tsne', n_components=2, interactive=False
        )
        assert fig is not None, "t-SNE 2D visualization failed"
        
        # Test PCA 2D
        fig = visualize_embedding_space(
            features, labels, method='pca', n_components=2, interactive=False
        )
        assert fig is not None, "PCA 2D visualization failed"
        
        # Test PCA 3D
        fig = visualize_embedding_space(
            features, labels, method='pca', n_components=3, interactive=False
        )
        assert fig is not None, "PCA 3D visualization failed"
        
        print("✓ visualize_embedding_space() passed")
        return True
    except Exception as e:
        print(f"✗ visualize_embedding_space() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualize_attention_maps():
    """Test attention map visualization."""
    print("\nTesting visualize_attention_maps()...")
    try:
        from visualizations import visualize_attention_maps
        
        # Test single-head attention
        attention_single = np.random.rand(20, 10)
        attention_single = attention_single / attention_single.sum(axis=1, keepdims=True)
        
        fig = visualize_attention_maps(attention_single)
        assert fig is not None, "Single-head attention visualization failed"
        
        # Test multi-head attention
        attention_multi = np.random.rand(4, 20, 10)
        attention_multi = attention_multi / attention_multi.sum(axis=2, keepdims=True)
        
        fig = visualize_attention_maps(attention_multi)
        assert fig is not None, "Multi-head attention visualization failed"
        
        # Test with torch tensor
        attention_tensor = torch.randn(20, 10).softmax(dim=-1)
        fig = visualize_attention_maps(attention_tensor)
        assert fig is not None, "Torch tensor attention visualization failed"
        
        print("✓ visualize_attention_maps() passed")
        return True
    except Exception as e:
        print(f"✗ visualize_attention_maps() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualize_weight_distributions():
    """Test weight distribution visualization."""
    print("\nTesting visualize_weight_distributions()...")
    try:
        from visualizations import visualize_weight_distributions
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.conv2 = nn.Conv2d(16, 32, 3)
                self.fc1 = nn.Linear(32, 64)
                self.fc2 = nn.Linear(64, 10)
        
        model = SimpleModel()
        
        # Test all layers
        fig = visualize_weight_distributions(model)
        assert fig is not None, "All layers visualization failed"
        
        # Test specific layer types
        fig = visualize_weight_distributions(model, layer_types=['Conv2d'])
        assert fig is not None, "Conv2d layers visualization failed"
        
        fig = visualize_weight_distributions(model, layer_types=['Linear'])
        assert fig is not None, "Linear layers visualization failed"
        
        print("✓ visualize_weight_distributions() passed")
        return True
    except Exception as e:
        print(f"✗ visualize_weight_distributions() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualize_feature_analysis():
    """Test feature analysis visualization."""
    print("\nTesting visualize_feature_analysis()...")
    try:
        from visualizations import visualize_feature_analysis
        
        # Create synthetic data with some interesting properties
        np.random.seed(42)
        n_samples = 200
        n_features = 100
        n_classes = 5
        
        features = []
        labels = []
        for i in range(n_classes):
            class_center = np.random.randn(n_features) * 2
            class_samples = class_center + np.random.randn(n_samples // n_classes, n_features) * 0.5
            features.append(class_samples)
            labels.extend([i] * (n_samples // n_classes))
        
        features = np.vstack(features)
        labels = np.array(labels)
        
        fig = visualize_feature_analysis(features, labels)
        assert fig is not None, "Feature analysis visualization failed"
        
        print("✓ visualize_feature_analysis() passed")
        return True
    except Exception as e:
        print(f"✗ visualize_feature_analysis() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhance_feature_visualizer():
    """Test FeatureVisualizer enhancement."""
    print("\nTesting enhance_feature_visualizer()...")
    try:
        from visualizations import enhance_feature_visualizer
        from feature_visualizer import FeatureVisualizer
        
        # Enhance the class
        enhance_feature_visualizer()
        
        # Check that new methods are available
        assert hasattr(FeatureVisualizer, 'visualize_embedding_space'), \
            "visualize_embedding_space method not added"
        assert hasattr(FeatureVisualizer, 'visualize_feature_analysis'), \
            "visualize_feature_analysis method not added"
        
        print("✓ enhance_feature_visualizer() passed")
        return True
    except Exception as e:
        print(f"✗ enhance_feature_visualizer() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("="*60)
    print("Running Visualization Module Tests")
    print("="*60)
    
    tests = [
        test_imports,
        test_visualize_embedding_space,
        test_visualize_attention_maps,
        test_visualize_weight_distributions,
        test_visualize_feature_analysis,
        test_enhance_feature_visualizer
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    
    exit_code = run_all_tests()
    
    # Close all figures to avoid warnings
    plt.close('all')
    
    sys.exit(exit_code)
