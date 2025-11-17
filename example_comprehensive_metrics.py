"""
Example: Using Comprehensive Evaluation Metrics

This script demonstrates how to use the new comprehensive evaluation
features for few-shot learning models.

Author: Comprehensive evaluation implementation
"""

import numpy as np
from feature_analysis import (
    compute_confidence_interval,
    detect_feature_collapse,
    compute_feature_utilization,
    compute_diversity_score,
    analyze_feature_redundancy,
    compute_intraclass_consistency,
    identify_confusing_pairs,
    compute_imbalance_ratio,
    comprehensive_feature_analysis
)


def example_confidence_interval():
    """Example: Computing 95% confidence intervals."""
    print("\n" + "="*80)
    print("Example 1: 95% Confidence Interval")
    print("="*80)
    print("\nProblem: We ran 1000 test episodes and got varying accuracies.")
    print("We want to estimate the true performance with confidence.")
    
    # Simulate 1000 episode accuracies
    np.random.seed(42)
    episode_accuracies = np.random.beta(8, 2, 1000)  # Beta distribution simulating accuracies
    
    print(f"\nEpisodes tested: {len(episode_accuracies)}")
    print(f"Raw mean accuracy: {np.mean(episode_accuracies):.4f}")
    
    # Compute confidence interval
    mean, lower, upper = compute_confidence_interval(episode_accuracies, confidence=0.95)
    
    print(f"\n📊 Results:")
    print(f"  Mean accuracy: {mean:.4f} ({mean*100:.2f}%)")
    print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
    print(f"  Margin: ±{(mean-lower):.4f} (±{(mean-lower)*100:.2f}%)")
    print(f"\nInterpretation: We are 95% confident that the true model")
    print(f"performance lies between {lower*100:.2f}% and {upper*100:.2f}%")


def example_feature_collapse():
    """Example: Detecting feature collapse."""
    print("\n" + "="*80)
    print("Example 2: Feature Collapse Detection")
    print("="*80)
    print("\nProblem: Some feature dimensions may have 'collapsed' (very low variance)")
    print("indicating they don't contribute useful information.")
    
    # Create synthetic features with some collapsed dimensions
    np.random.seed(42)
    n_samples = 200
    n_features = 128
    
    features = np.random.randn(n_samples, n_features)
    # Make some dimensions collapse
    features[:, 50:60] = np.random.randn(n_samples, 10) * 1e-5  # Very low variance
    features[:, 100:105] = 0.5  # Constant values
    
    print(f"\nFeature matrix shape: {features.shape}")
    
    # Detect collapse
    result = detect_feature_collapse(features, threshold=1e-4)
    
    print(f"\n📊 Results:")
    print(f"  Total dimensions: {result['total_dimensions']}")
    print(f"  Collapsed dimensions: {result['collapsed_dimensions']}")
    print(f"  Collapse ratio: {result['collapse_ratio']*100:.2f}%")
    print(f"  Std range: [{result['min_std']:.6f}, {result['max_std']:.6f}]")
    print(f"\nInterpretation: {result['collapsed_dimensions']} dimensions have")
    print(f"effectively collapsed and provide minimal discriminative information.")


def example_feature_redundancy():
    """Example: Analyzing feature redundancy."""
    print("\n" + "="*80)
    print("Example 3: Feature Redundancy Analysis")
    print("="*80)
    print("\nProblem: Multiple features may carry similar information (redundancy)")
    print("indicating we could reduce dimensionality without losing information.")
    
    # Create features with some redundancy
    np.random.seed(42)
    n_samples = 300
    base_features = np.random.randn(n_samples, 40)
    
    # Add redundant features (correlated copies with noise)
    redundant_features = base_features[:, :20] + np.random.randn(n_samples, 20) * 0.1
    features = np.concatenate([base_features, redundant_features], axis=1)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"(Created {redundant_features.shape[1]} intentionally redundant features)")
    
    # Analyze redundancy
    result = analyze_feature_redundancy(features)
    
    print(f"\n📊 Results:")
    print(f"  Total features: {result['total_features']}")
    print(f"  Effective dimensions (95% variance): {result['effective_dimensions_95pct']}")
    print(f"  Dimensionality reduction ratio: {result['dimensionality_reduction_ratio']:.4f}")
    print(f"  High correlation pairs (>0.9): {result['high_correlation_pairs']}")
    print(f"  Moderate correlation pairs (>0.7): {result['moderate_correlation_pairs']}")
    print(f"\nInterpretation: We can reduce from {result['total_features']} to")
    print(f"{result['effective_dimensions_95pct']} dimensions while retaining 95% of variance.")


def example_intraclass_consistency():
    """Example: Computing intra-class consistency."""
    print("\n" + "="*80)
    print("Example 4: Intra-Class Consistency")
    print("="*80)
    print("\nProblem: We want to measure how uniformly the model represents")
    print("samples from the same class (high consistency = tight clusters).")
    
    # Create features with varying consistency across classes
    np.random.seed(42)
    n_per_class = 50
    n_features = 64
    n_classes = 5
    
    features = []
    labels = []
    
    for class_id in range(n_classes):
        # Each class has different spread
        spread = 0.5 + class_id * 0.3  # Increasing spread
        class_features = np.random.randn(n_per_class, n_features) * spread
        class_features += class_id * 3  # Separate centroids
        
        features.append(class_features)
        labels.extend([class_id] * n_per_class)
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    print(f"\nFeatures: {features.shape}, Classes: {n_classes}")
    
    # Compute consistency
    result = compute_intraclass_consistency(features, labels)
    
    print(f"\n📊 Results:")
    print(f"  Mean Euclidean consistency: {result['mean_euclidean_consistency']:.4f}")
    print(f"  Mean Cosine consistency: {result['mean_cosine_consistency']:.4f}")
    print(f"  Mean Combined consistency: {result['mean_combined_consistency']:.4f}")
    print(f"\nPer-class combined consistency:")
    for i, score in enumerate(result['per_class_combined']):
        print(f"    Class {i}: {score:.4f}")
    print(f"\nInterpretation: Higher scores indicate more consistent (tighter)")
    print(f"representations within each class.")


def example_confusing_pairs():
    """Example: Identifying confusing class pairs."""
    print("\n" + "="*80)
    print("Example 5: Confusing Class Pairs")
    print("="*80)
    print("\nProblem: Some classes may be hard to distinguish because their")
    print("feature representations are very similar (close centroids).")
    
    # Create features with some overlapping classes
    np.random.seed(42)
    n_per_class = 100
    n_features = 64
    class_names = ['Dog', 'Cat', 'Wolf', 'Tiger', 'Lion']
    
    features = []
    labels = []
    
    # Dog and Wolf are close (confusing)
    dog_features = np.random.randn(n_per_class, n_features) * 0.8
    wolf_features = dog_features + np.random.randn(n_per_class, n_features) * 0.5
    
    # Cat, Tiger, Lion - Cat close to Tiger
    cat_features = np.random.randn(n_per_class, n_features) * 0.8 + 5
    tiger_features = cat_features + np.random.randn(n_per_class, n_features) * 0.6
    lion_features = tiger_features + np.random.randn(n_per_class, n_features) * 2
    
    features = np.vstack([dog_features, cat_features, wolf_features, tiger_features, lion_features])
    labels = np.repeat(np.arange(5), n_per_class)
    
    print(f"\nFeatures: {features.shape}, Classes: {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}")
    
    # Identify confusing pairs
    result = identify_confusing_pairs(features, labels, top_k=3)
    
    print(f"\n📊 Most Confusing Pairs (closest centroids):")
    for i, pair in enumerate(result['most_confusing_pairs'], 1):
        name1 = class_names[pair['class_1']]
        name2 = class_names[pair['class_2']]
        print(f"  {i}. {name1} ↔ {name2}: distance = {pair['distance']:.4f}")
    
    print(f"\n  Mean inter-centroid distance: {result['mean_intercentroid_distance']:.4f}")
    print(f"\nInterpretation: Classes with smaller distances are more likely")
    print(f"to be confused by the model (harder to distinguish).")


def example_comprehensive_analysis():
    """Example: Full comprehensive analysis."""
    print("\n" + "="*80)
    print("Example 6: Comprehensive Feature Analysis")
    print("="*80)
    print("\nProblem: We want to understand all aspects of our feature space")
    print("in a single comprehensive report.")
    
    # Create realistic feature scenario
    np.random.seed(42)
    n_samples = 500
    n_features = 128
    n_classes = 10
    
    # Generate features with various characteristics
    features = np.random.randn(n_samples, n_features)
    
    # Add class-specific patterns
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    for i in range(n_classes):
        mask = labels == i
        features[mask, :20] += i * 1.5
        features[mask, 20:40] *= (1 + i * 0.2)
    
    # Add some collapsed dimensions
    features[:, 100:110] *= 1e-5
    
    print(f"\nRunning comprehensive analysis...")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    
    # Run comprehensive analysis
    results = comprehensive_feature_analysis(features, labels)
    
    print(f"\n📊 Comprehensive Results:")
    print(f"\n1. Feature Collapse:")
    fc = results['feature_collapse']
    print(f"   Collapsed: {fc['collapsed_dimensions']}/{fc['total_dimensions']} ({fc['collapse_ratio']*100:.1f}%)")
    
    print(f"\n2. Feature Utilization:")
    fu = results['feature_utilization']
    print(f"   Mean: {fu['mean_utilization']:.4f}")
    print(f"   Low utilization dims: {fu['low_utilization_dims']}")
    
    print(f"\n3. Diversity:")
    ds = results['diversity_score']
    print(f"   Mean diversity: {ds['mean_diversity']:.4f}")
    
    print(f"\n4. Redundancy:")
    fr = results['feature_redundancy']
    print(f"   Effective dims: {fr['effective_dimensions_95pct']}/{fr['total_features']}")
    print(f"   High corr pairs: {fr['high_correlation_pairs']}")
    
    print(f"\n5. Intra-class Consistency:")
    ic = results['intraclass_consistency']
    print(f"   Combined: {ic['mean_combined_consistency']:.4f}")
    
    print(f"\n6. Confusing Pairs:")
    cp = results['confusing_pairs']
    print(f"   Closest pair distance: {cp['most_confusing_pairs'][0]['distance']:.4f}")
    
    print(f"\n7. Class Imbalance:")
    ir = results['imbalance_ratio']
    print(f"   Ratio: {ir['imbalance_ratio']:.4f}")
    
    print(f"\nInterpretation: This comprehensive view helps identify")
    print(f"potential issues and optimization opportunities in the feature space.")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION METRICS - EXAMPLES & USAGE")
    print("="*80)
    print("\nThese examples demonstrate how to use the new evaluation features")
    print("for analyzing few-shot learning models.")
    
    # Run examples
    example_confidence_interval()
    example_feature_collapse()
    example_feature_redundancy()
    example_intraclass_consistency()
    example_confusing_pairs()
    example_comprehensive_analysis()
    
    # Summary
    print("\n" + "="*80)
    print("USAGE IN PRACTICE")
    print("="*80)
    print("\nTo use these features in your evaluation:")
    print("\n1. Standard comprehensive evaluation (includes CI, F1, confusion matrix):")
    print("   python test.py --dataset miniImagenet --comprehensive_eval 1")
    print("\n2. With full feature analysis:")
    print("   python test.py --dataset miniImagenet --feature_analysis 1")
    print("\n3. For ablation studies, see ABLATION_STUDIES.md")
    print("\nAll metrics will be displayed in the output and can guide")
    print("model improvements and architectural decisions.")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
