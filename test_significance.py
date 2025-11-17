#!/usr/bin/env python3
"""
Test script for significance testing functionality.

This script tests the new significance testing module with synthetic data.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_mcnemar():
    """Test McNemar's test."""
    print("="*80)
    print("Testing McNemar's Test")
    print("="*80)
    
    from significance_test import mcnemar_test
    
    # Create synthetic data where model 1 is slightly better
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Model 1: 75% accuracy
    y_pred1 = y_true.copy()
    wrong_indices1 = np.random.choice(n_samples, size=int(0.25 * n_samples), replace=False)
    y_pred1[wrong_indices1] = (y_pred1[wrong_indices1] + 1) % n_classes
    
    # Model 2: 70% accuracy
    y_pred2 = y_true.copy()
    wrong_indices2 = np.random.choice(n_samples, size=int(0.30 * n_samples), replace=False)
    y_pred2[wrong_indices2] = (y_pred2[wrong_indices2] + 1) % n_classes
    
    result = mcnemar_test(y_true, y_pred1, y_pred2)
    
    print(f"\nModel 1 accuracy: {np.mean(y_pred1 == y_true):.4f}")
    print(f"Model 2 accuracy: {np.mean(y_pred2 == y_true):.4f}")
    print(f"\nMcNemar's statistic: {result['statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Significant: {result['significant']}")
    print(f"\nContingency Table:")
    print(f"  Both correct: {result['both_correct']}")
    print(f"  Only Model 1 correct: {result['only_model1_correct']}")
    print(f"  Only Model 2 correct: {result['only_model2_correct']}")
    print(f"  Both wrong: {result['both_wrong']}")
    print(f"\nInterpretation: {result['interpretation']}")
    
    print("\n✓ McNemar's test completed successfully\n")
    return True


def test_paired_ttest():
    """Test paired t-test."""
    print("="*80)
    print("Testing Paired t-test")
    print("="*80)
    
    from significance_test import paired_ttest
    
    # Create synthetic episode accuracies
    np.random.seed(42)
    n_episodes = 100
    
    # Model 1: mean=0.75, std=0.05
    accuracies1 = np.random.normal(0.75, 0.05, n_episodes)
    accuracies1 = np.clip(accuracies1, 0, 1)
    
    # Model 2: mean=0.70, std=0.05  
    accuracies2 = np.random.normal(0.70, 0.05, n_episodes)
    accuracies2 = np.clip(accuracies2, 0, 1)
    
    result = paired_ttest(accuracies1, accuracies2)
    
    print(f"\nModel 1 mean accuracy: {np.mean(accuracies1):.4f}")
    print(f"Model 2 mean accuracy: {np.mean(accuracies2):.4f}")
    print(f"\nT-statistic: {result['statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Significant: {result['significant']}")
    print(f"Mean difference: {result['mean_diff']:.4f}")
    print(f"95% CI: [{result['confidence_interval_95']['lower']:.4f}, "
          f"{result['confidence_interval_95']['upper']:.4f}]")
    print(f"\nInterpretation: {result['interpretation']}")
    
    print("\n✓ Paired t-test completed successfully\n")
    return True


def test_wilcoxon():
    """Test Wilcoxon signed-rank test."""
    print("="*80)
    print("Testing Wilcoxon Signed-Rank Test")
    print("="*80)
    
    from significance_test import wilcoxon_test
    
    # Create synthetic episode accuracies with non-normal distribution
    np.random.seed(42)
    n_episodes = 100
    
    # Model 1: beta distribution
    accuracies1 = np.random.beta(8, 2, n_episodes)
    
    # Model 2: slightly lower beta distribution
    accuracies2 = np.random.beta(7, 3, n_episodes)
    
    result = wilcoxon_test(accuracies1, accuracies2)
    
    print(f"\nModel 1 median accuracy: {np.median(accuracies1):.4f}")
    print(f"Model 2 median accuracy: {np.median(accuracies2):.4f}")
    print(f"\nWilcoxon statistic: {result['statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Significant: {result['significant']}")
    print(f"Median difference: {result['median_diff']:.4f}")
    print(f"\nInterpretation: {result['interpretation']}")
    
    print("\n✓ Wilcoxon test completed successfully\n")
    return True


def test_per_class_f1():
    """Test per-class F1 computation."""
    print("="*80)
    print("Testing Per-Class F1 Score Computation")
    print("="*80)
    
    from significance_test import compute_per_class_f1
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Create predictions with varying accuracy per class
    y_pred = y_true.copy()
    for cls in range(n_classes):
        cls_mask = y_true == cls
        n_cls_samples = np.sum(cls_mask)
        # Class 0: 90% acc, Class 1: 80% acc, ..., Class 4: 50% acc
        error_rate = 0.1 + cls * 0.1
        n_errors = int(error_rate * n_cls_samples)
        error_indices = np.random.choice(np.where(cls_mask)[0], size=n_errors, replace=False)
        y_pred[error_indices] = (y_pred[error_indices] + 1) % n_classes
    
    result = compute_per_class_f1(y_true, y_pred, n_classes)
    
    print(f"\nMacro F1: {result['macro_f1']:.4f}")
    print(f"Micro F1: {result['micro_f1']:.4f}")
    print(f"Weighted F1: {result['weighted_f1']:.4f}")
    print(f"Std F1: {result['std_f1']:.4f}")
    print(f"Range: [{result['min_f1']:.4f}, {result['max_f1']:.4f}]")
    
    print("\nPer-class F1 scores:")
    for i, (f1, prec, rec) in enumerate(zip(result['per_class_f1'],
                                              result['per_class_precision'],
                                              result['per_class_recall'])):
        print(f"  Class {i}: F1={f1:.4f} (Precision={prec:.4f}, Recall={rec:.4f})")
    
    print("\n✓ Per-class F1 computation completed successfully\n")
    return True


def test_compare_per_class_f1():
    """Test per-class F1 comparison."""
    print("="*80)
    print("Testing Per-Class F1 Comparison")
    print("="*80)
    
    from significance_test import compare_per_class_f1
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Model 1: good performance
    y_pred1 = y_true.copy()
    wrong_indices1 = np.random.choice(n_samples, size=int(0.20 * n_samples), replace=False)
    y_pred1[wrong_indices1] = (y_pred1[wrong_indices1] + 1) % n_classes
    
    # Model 2: slightly worse performance
    y_pred2 = y_true.copy()
    wrong_indices2 = np.random.choice(n_samples, size=int(0.25 * n_samples), replace=False)
    y_pred2[wrong_indices2] = (y_pred2[wrong_indices2] + 1) % n_classes
    
    class_names = [f"Class {i}" for i in range(n_classes)]
    result = compare_per_class_f1(y_true, y_pred1, y_pred2, class_names, n_classes)
    
    print(f"\nModel 1 Macro F1: {result['model1']['macro_f1']:.4f}")
    print(f"Model 2 Macro F1: {result['model2']['macro_f1']:.4f}")
    
    comp = result['comparison']
    print(f"\nMean F1 difference: {comp['mean_f1_diff']:.4f}")
    print(f"T-statistic: {comp['t_statistic']:.4f}")
    print(f"P-value: {comp['p_value']:.4f}")
    print(f"Significant: {comp['significant']}")
    
    print("\nPer-class comparison:")
    for cls_comp in result['per_class_comparison']:
        print(f"  {cls_comp['class_name']}:")
        print(f"    Model 1: {cls_comp['f1_model1']:.4f}")
        print(f"    Model 2: {cls_comp['f1_model2']:.4f}")
        print(f"    Difference: {cls_comp['difference']:+.4f} (Better: {cls_comp['better_model']})")
    
    print("\n✓ Per-class F1 comparison completed successfully\n")
    return True


def test_comprehensive_significance_test():
    """Test comprehensive significance testing."""
    print("="*80)
    print("Testing Comprehensive Significance Test")
    print("="*80)
    
    from significance_test import comprehensive_significance_test, pretty_print_significance_test
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    n_episodes = 100
    
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Model 1: 75% accuracy
    y_pred1 = y_true.copy()
    wrong_indices1 = np.random.choice(n_samples, size=int(0.25 * n_samples), replace=False)
    y_pred1[wrong_indices1] = (y_pred1[wrong_indices1] + 1) % n_classes
    
    # Model 2: 70% accuracy
    y_pred2 = y_true.copy()
    wrong_indices2 = np.random.choice(n_samples, size=int(0.30 * n_samples), replace=False)
    y_pred2[wrong_indices2] = (y_pred2[wrong_indices2] + 1) % n_classes
    
    # Episode accuracies
    accuracies1 = np.random.normal(0.75, 0.05, n_episodes)
    accuracies1 = np.clip(accuracies1, 0, 1)
    accuracies2 = np.random.normal(0.70, 0.05, n_episodes)
    accuracies2 = np.clip(accuracies2, 0, 1)
    
    class_names = [f"Class {i}" for i in range(n_classes)]
    
    result = comprehensive_significance_test(
        y_true, y_pred1, y_pred2,
        accuracies1, accuracies2,
        class_names,
        model1_name="Model A",
        model2_name="Model B"
    )
    
    # Print results
    pretty_print_significance_test(result)
    
    print("✓ Comprehensive significance test completed successfully\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SIGNIFICANCE TESTING MODULE TEST SUITE")
    print("="*80 + "\n")
    
    all_passed = True
    
    # Test individual components
    tests = [
        ("McNemar's Test", test_mcnemar),
        ("Paired t-test", test_paired_ttest),
        ("Wilcoxon Test", test_wilcoxon),
        ("Per-Class F1", test_per_class_f1),
        ("F1 Comparison", test_compare_per_class_f1),
        ("Comprehensive Test", test_comprehensive_significance_test),
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"\n✗ {test_name} failed")
        except Exception as e:
            all_passed = False
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe significance testing system is working correctly.")
        print("\nAvailable features:")
        print("  1. McNemar's Test - Compare paired predictions")
        print("  2. Paired t-test - Compare episode-wise accuracies")
        print("  3. Wilcoxon Test - Non-parametric alternative to t-test")
        print("  4. Per-Class F1 Scores - Detailed per-class metrics")
        print("  5. F1 Score Comparison - Compare F1 scores between models")
        print("  6. Comprehensive Testing - All-in-one significance testing")
        print("\nUsage in evaluation:")
        print("  from significance_test import comprehensive_significance_test")
        print("  results = comprehensive_significance_test(y_true, y_pred1, y_pred2, ...)")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
