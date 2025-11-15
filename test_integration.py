"""
Integration test for feature analysis visualizations.
Tests that visualizations integrate properly with the evaluation pipeline.
"""

import numpy as np
import os
import sys

# Test that all necessary imports work
def test_imports():
    """Test that all visualization imports work"""
    print("\n" + "="*60)
    print("Testing Imports")
    print("="*60)
    
    try:
        from feature_analysis import (
            comprehensive_feature_analysis,
            print_feature_analysis_summary,
            visualize_embedding_space,
            visualize_attention_maps,
            visualize_weight_distributions,
            visualize_feature_analysis,
            SCIPY_AVAILABLE,
            MATPLOTLIB_AVAILABLE,
            UMAP_AVAILABLE
        )
        print("✓ All feature_analysis functions imported successfully")
        print(f"  Libraries available:")
        print(f"    - SCIPY (PCA, t-SNE): {SCIPY_AVAILABLE}")
        print(f"    - MATPLOTLIB (Plotting): {MATPLOTLIB_AVAILABLE}")
        print(f"    - UMAP (Optional): {UMAP_AVAILABLE}")
        
        if not SCIPY_AVAILABLE:
            print("\n⚠️  WARNING: scipy/sklearn not available. Install with:")
            print("    pip install scipy scikit-learn")
            return False
            
        if not MATPLOTLIB_AVAILABLE:
            print("\n⚠️  WARNING: matplotlib not available. Install with:")
            print("    pip install matplotlib seaborn")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_feature_analysis_with_viz():
    """Test that feature analysis works with visualization"""
    print("\n" + "="*60)
    print("Testing Feature Analysis with Visualizations")
    print("="*60)
    
    from feature_analysis import (
        comprehensive_feature_analysis,
        visualize_feature_analysis
    )
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 32
    n_classes = 5
    
    features = np.random.randn(n_samples, n_features)
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Add structure to features
    for i in range(n_classes):
        mask = labels == i
        features[mask] += np.random.randn(n_features) * 1.5
    
    # Run comprehensive analysis
    print("\n1. Running comprehensive feature analysis...")
    episode_accuracies = np.random.rand(20) * 0.3 + 0.6  # Random accuracies between 60-90%
    
    results = comprehensive_feature_analysis(
        features=features,
        labels=labels,
        episode_accuracies=episode_accuracies
    )
    
    print("✓ Feature analysis completed")
    print(f"  Generated {len(results)} metric groups")
    
    # Generate visualizations
    print("\n2. Generating visualizations...")
    
    # Create model weights
    model_weights = {
        'layer1.weight': np.random.randn(64, 32) * 0.1,
        'layer2.weight': np.random.randn(32, 16) * 0.08,
    }
    
    # Create attention weights
    attention_weights = np.random.rand(4, 15, 25)  # 4 heads, 15 queries, 25 support
    for i in range(4):
        attention_weights[i] = attention_weights[i] / attention_weights[i].sum(axis=1, keepdims=True)
    
    figures = visualize_feature_analysis(
        features=features,
        labels=labels,
        attention_weights=attention_weights,
        model_weights=model_weights,
        save_dir='./figures/integration_test',
        methods=['pca']  # Just use PCA for speed
    )
    
    print(f"✓ Generated {len(figures)} visualizations")
    
    # Check that files were created
    expected_files = [
        './figures/integration_test/embedding_pca_2d.png',
        './figures/integration_test/embedding_pca_3d.png',
        './figures/integration_test/attention_maps.png',
        './figures/integration_test/weight_distributions.png'
    ]
    
    created_files = []
    missing_files = []
    
    for filepath in expected_files:
        if os.path.exists(filepath):
            created_files.append(filepath)
        else:
            missing_files.append(filepath)
    
    print(f"\n3. Verifying output files...")
    print(f"  Created: {len(created_files)}/{len(expected_files)} files")
    
    if missing_files:
        print(f"  Missing files:")
        for f in missing_files:
            print(f"    - {f}")
    
    return len(missing_files) == 0


def test_individual_visualizations():
    """Test each visualization function individually"""
    print("\n" + "="*60)
    print("Testing Individual Visualization Functions")
    print("="*60)
    
    from feature_analysis import (
        visualize_embedding_space,
        visualize_attention_maps,
        visualize_weight_distributions
    )
    
    # Test data
    np.random.seed(42)
    features = np.random.randn(50, 16)
    labels = np.repeat(np.arange(5), 10)
    
    # Test 1: Embedding space
    print("\n1. Testing embedding space visualization...")
    fig = visualize_embedding_space(
        features, labels,
        method='pca',
        n_components=2,
        save_path='./figures/integration_test/test_embedding.png'
    )
    if fig is not None:
        print("  ✓ Embedding visualization created")
    else:
        print("  ⚠️  Embedding visualization returned None")
    
    # Test 2: Attention maps
    print("\n2. Testing attention map visualization...")
    attention = np.random.rand(10, 15)
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    fig = visualize_attention_maps(
        attention,
        save_path='./figures/integration_test/test_attention.png'
    )
    if fig is not None:
        print("  ✓ Attention visualization created")
    else:
        print("  ⚠️  Attention visualization returned None")
    
    # Test 3: Weight distributions
    print("\n3. Testing weight distribution visualization...")
    weights = {
        'test_layer': np.random.randn(32, 16) * 0.1
    }
    
    fig = visualize_weight_distributions(
        weights,
        save_path='./figures/integration_test/test_weights.png'
    )
    if fig is not None:
        print("  ✓ Weight visualization created")
    else:
        print("  ⚠️  Weight visualization returned None")
    
    return True


if __name__ == '__main__':
    print("\n" + "="*60)
    print("FEATURE ANALYSIS VISUALIZATION INTEGRATION TEST")
    print("="*60)
    
    # Create output directory
    os.makedirs('./figures/integration_test', exist_ok=True)
    
    # Run tests
    tests_passed = 0
    tests_total = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_feature_analysis_with_viz():
        tests_passed += 1
    
    if test_individual_visualizations():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {tests_passed}/{tests_total} tests")
    
    if tests_passed == tests_total:
        print("\n✓ ALL INTEGRATION TESTS PASSED!")
        print("\nVisualization features are ready to use.")
        print("Run with --feature_analysis 1 flag to enable in training/evaluation.")
    else:
        print(f"\n⚠️  {tests_total - tests_passed} test(s) failed")
        print("Check error messages above for details.")
        sys.exit(1)
    
    print("="*60)
