"""
Statistical Significance Testing for Few-Shot Learning

This module provides statistical tests to compare the performance of
different models or methods, including:
- McNemar's test for paired predictions
- Paired t-test for episode-wise accuracies
- Wilcoxon signed-rank test (non-parametric alternative)
- Per-class F1 score comparison with significance tests

Author: Significance testing implementation
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def mcnemar_test(y_true: np.ndarray, 
                 y_pred1: np.ndarray, 
                 y_pred2: np.ndarray,
                 continuity: bool = True) -> Dict:
    """
    Perform McNemar's test to compare two models on the same test set.
    
    McNemar's test is appropriate for comparing paired nominal data.
    It tests whether the disagreements between two models are significantly
    different from each other.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        continuity: Whether to apply continuity correction
    
    Returns:
        Dictionary with test results including:
        - statistic: McNemar's test statistic
        - p_value: p-value of the test
        - significant: Whether the difference is significant at α=0.05
        - contingency_table: 2x2 contingency table
    """
    # Create contingency table
    # Table format:
    #              Model 2 Correct | Model 2 Wrong
    # Model 1 Correct    n00       |     n01
    # Model 1 Wrong      n10       |     n11
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    n00 = np.sum(correct1 & correct2)   # Both correct
    n01 = np.sum(correct1 & ~correct2)  # Only model 1 correct
    n10 = np.sum(~correct1 & correct2)  # Only model 2 correct
    n11 = np.sum(~correct1 & ~correct2) # Both wrong
    
    contingency_table = np.array([[n00, n01], [n10, n11]])
    
    # McNemar's test focuses on the disagreements (n01 and n10)
    b = n01
    c = n10
    
    # Apply continuity correction if requested
    if continuity:
        statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    else:
        statistic = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0
    
    # Chi-square distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(statistic, 1)
    
    return {
        'test_name': "McNemar's Test",
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'alpha': 0.05,
        'contingency_table': contingency_table.tolist(),
        'both_correct': int(n00),
        'only_model1_correct': int(n01),
        'only_model2_correct': int(n10),
        'both_wrong': int(n11),
        'interpretation': (
            f"Model 1 wins {n01} cases, Model 2 wins {n10} cases. "
            f"{'Significant' if p_value < 0.05 else 'No significant'} difference (p={p_value:.4f})."
        )
    }


def paired_ttest(accuracies1: np.ndarray, 
                 accuracies2: np.ndarray,
                 alternative: str = 'two-sided') -> Dict:
    """
    Perform paired t-test to compare episode-wise accuracies of two models.
    
    Args:
        accuracies1: Array of episode accuracies from model 1
        accuracies2: Array of episode accuracies from model 2
        alternative: Type of test ('two-sided', 'less', 'greater')
    
    Returns:
        Dictionary with test results including:
        - statistic: t-statistic
        - p_value: p-value of the test
        - significant: Whether the difference is significant at α=0.05
        - mean_diff: Mean difference between models
        - confidence_interval: 95% confidence interval of the difference
    """
    if len(accuracies1) != len(accuracies2):
        raise ValueError("Arrays must have the same length")
    
    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(accuracies1, accuracies2, alternative=alternative)
    
    # Compute mean difference and confidence interval
    differences = accuracies1 - accuracies2
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    
    # 95% confidence interval
    ci_margin = stats.t.ppf(0.975, n - 1) * std_diff / np.sqrt(n)
    ci_lower = mean_diff - ci_margin
    ci_upper = mean_diff + ci_margin
    
    return {
        'test_name': 'Paired t-test',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'alpha': 0.05,
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'confidence_interval_95': {
            'lower': float(ci_lower),
            'upper': float(ci_upper),
            'margin': float(ci_margin)
        },
        'n_episodes': int(n),
        'interpretation': (
            f"Mean difference: {mean_diff:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
            f"{'Significant' if p_value < 0.05 else 'No significant'} difference (p={p_value:.4f})."
        )
    }


def wilcoxon_test(accuracies1: np.ndarray,
                  accuracies2: np.ndarray,
                  alternative: str = 'two-sided') -> Dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    This test is more robust when the normality assumption is violated.
    
    Args:
        accuracies1: Array of episode accuracies from model 1
        accuracies2: Array of episode accuracies from model 2
        alternative: Type of test ('two-sided', 'less', 'greater')
    
    Returns:
        Dictionary with test results
    """
    if len(accuracies1) != len(accuracies2):
        raise ValueError("Arrays must have the same length")
    
    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(accuracies1, accuracies2, 
                                        alternative=alternative, 
                                        zero_method='wilcox')
    
    # Compute median difference
    differences = accuracies1 - accuracies2
    median_diff = np.median(differences)
    
    return {
        'test_name': 'Wilcoxon Signed-Rank Test',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'alpha': 0.05,
        'median_diff': float(median_diff),
        'mean_diff': float(np.mean(differences)),
        'interpretation': (
            f"Median difference: {median_diff:.4f}. "
            f"{'Significant' if p_value < 0.05 else 'No significant'} difference (p={p_value:.4f})."
        )
    }


def compute_per_class_f1(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         n_classes: Optional[int] = None) -> Dict:
    """
    Compute F1 score for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes (if None, inferred from labels)
    
    Returns:
        Dictionary with per-class F1 scores and statistics
    """
    if n_classes is None:
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    f1_scores = []
    precisions = []
    recalls = []
    
    for cls in range(n_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    
    f1_scores = np.array(f1_scores)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    return {
        'per_class_f1': f1_scores.tolist(),
        'per_class_precision': precisions.tolist(),
        'per_class_recall': recalls.tolist(),
        'macro_f1': float(np.mean(f1_scores)),
        'micro_f1': compute_micro_f1(y_true, y_pred),
        'weighted_f1': compute_weighted_f1(y_true, y_pred, f1_scores),
        'std_f1': float(np.std(f1_scores)),
        'min_f1': float(np.min(f1_scores)),
        'max_f1': float(np.max(f1_scores)),
        'n_classes': int(n_classes)
    }


def compute_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute micro-averaged F1 score (equivalent to accuracy)."""
    return float(np.mean(y_true == y_pred))


def compute_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, f1_scores: np.ndarray) -> float:
    """Compute weighted F1 score based on class support."""
    unique_classes, counts = np.unique(y_true, return_counts=True)
    weights = counts / len(y_true)
    
    # Ensure f1_scores has the same length as unique_classes
    if len(f1_scores) > len(unique_classes):
        f1_scores = f1_scores[:len(unique_classes)]
    
    return float(np.sum(f1_scores * weights))


def compare_per_class_f1(y_true: np.ndarray,
                         y_pred1: np.ndarray,
                         y_pred2: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         n_classes: Optional[int] = None) -> Dict:
    """
    Compare per-class F1 scores between two models with significance testing.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        class_names: Optional list of class names
        n_classes: Number of classes (if None, inferred from labels)
    
    Returns:
        Dictionary with comparison results
    """
    # Compute per-class F1 for both models
    f1_results1 = compute_per_class_f1(y_true, y_pred1, n_classes)
    f1_results2 = compute_per_class_f1(y_true, y_pred2, n_classes)
    
    n_classes = f1_results1['n_classes']
    f1_scores1 = np.array(f1_results1['per_class_f1'])
    f1_scores2 = np.array(f1_results2['per_class_f1'])
    
    # Compute differences
    f1_diff = f1_scores1 - f1_scores2
    
    # Paired t-test on F1 scores across classes
    if n_classes > 1:
        t_stat, p_value = stats.ttest_rel(f1_scores1, f1_scores2)
    else:
        t_stat, p_value = 0.0, 1.0
    
    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Per-class comparison
    class_comparisons = []
    for i in range(n_classes):
        class_comparisons.append({
            'class_name': class_names[i] if i < len(class_names) else f"Class {i}",
            'f1_model1': float(f1_scores1[i]),
            'f1_model2': float(f1_scores2[i]),
            'difference': float(f1_diff[i]),
            'better_model': 'Model 1' if f1_diff[i] > 0 else ('Model 2' if f1_diff[i] < 0 else 'Tie')
        })
    
    return {
        'model1': f1_results1,
        'model2': f1_results2,
        'comparison': {
            'mean_f1_diff': float(np.mean(f1_diff)),
            'std_f1_diff': float(np.std(f1_diff)),
            'max_improvement': float(np.max(f1_diff)),
            'max_degradation': float(np.min(f1_diff)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'alpha': 0.05
        },
        'per_class_comparison': class_comparisons
    }


def comprehensive_significance_test(y_true: np.ndarray,
                                    y_pred1: np.ndarray,
                                    y_pred2: np.ndarray,
                                    accuracies1: Optional[np.ndarray] = None,
                                    accuracies2: Optional[np.ndarray] = None,
                                    class_names: Optional[List[str]] = None,
                                    model1_name: str = "Model 1",
                                    model2_name: str = "Model 2") -> Dict:
    """
    Perform comprehensive significance testing between two models.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        accuracies1: Optional episode-wise accuracies for model 1
        accuracies2: Optional episode-wise accuracies for model 2
        class_names: Optional list of class names
        model1_name: Name of model 1
        model2_name: Name of model 2
    
    Returns:
        Dictionary with all test results
    """
    results = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'tests': {}
    }
    
    # McNemar's test
    results['tests']['mcnemar'] = mcnemar_test(y_true, y_pred1, y_pred2)
    
    # Per-class F1 comparison
    results['tests']['per_class_f1'] = compare_per_class_f1(
        y_true, y_pred1, y_pred2, class_names
    )
    
    # Episode-wise tests (if provided)
    if accuracies1 is not None and accuracies2 is not None:
        results['tests']['paired_ttest'] = paired_ttest(accuracies1, accuracies2)
        results['tests']['wilcoxon'] = wilcoxon_test(accuracies1, accuracies2)
    
    # Summary
    mcnemar_sig = results['tests']['mcnemar']['significant']
    f1_sig = results['tests']['per_class_f1']['comparison']['significant']
    
    results['summary'] = {
        'mcnemar_significant': mcnemar_sig,
        'per_class_f1_significant': f1_sig,
        'overall_significant': mcnemar_sig or f1_sig,
        'recommendation': (
            f"Based on McNemar's test (p={results['tests']['mcnemar']['p_value']:.4f}) "
            f"and per-class F1 comparison (p={results['tests']['per_class_f1']['comparison']['p_value']:.4f}), "
            f"{'there is a statistically significant difference' if (mcnemar_sig or f1_sig) else 'there is no statistically significant difference'} "
            f"between {model1_name} and {model2_name}."
        )
    }
    
    return results


def pretty_print_significance_test(results: Dict) -> None:
    """
    Print significance test results in a readable format.
    
    Args:
        results: Results from comprehensive_significance_test
    """
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TEST RESULTS")
    print("="*80)
    print(f"\nComparing: {results['model1_name']} vs {results['model2_name']}")
    
    # McNemar's Test
    if 'mcnemar' in results['tests']:
        print("\n" + "-"*80)
        print("McNemar's Test (Paired Predictions)")
        print("-"*80)
        mcn = results['tests']['mcnemar']
        print(f"Statistic: {mcn['statistic']:.4f}")
        print(f"P-value: {mcn['p_value']:.4f}")
        print(f"Significant (α=0.05): {'YES' if mcn['significant'] else 'NO'}")
        print(f"\nContingency Table:")
        print(f"  Both correct: {mcn['both_correct']}")
        print(f"  Only {results['model1_name']} correct: {mcn['only_model1_correct']}")
        print(f"  Only {results['model2_name']} correct: {mcn['only_model2_correct']}")
        print(f"  Both wrong: {mcn['both_wrong']}")
        print(f"\nInterpretation: {mcn['interpretation']}")
    
    # Per-class F1 Comparison
    if 'per_class_f1' in results['tests']:
        print("\n" + "-"*80)
        print("Per-Class F1 Score Comparison")
        print("-"*80)
        f1_comp = results['tests']['per_class_f1']
        
        print(f"\n{results['model1_name']}:")
        print(f"  Macro F1: {f1_comp['model1']['macro_f1']:.4f}")
        print(f"  Micro F1: {f1_comp['model1']['micro_f1']:.4f}")
        print(f"  Std F1: {f1_comp['model1']['std_f1']:.4f}")
        
        print(f"\n{results['model2_name']}:")
        print(f"  Macro F1: {f1_comp['model2']['macro_f1']:.4f}")
        print(f"  Micro F1: {f1_comp['model2']['micro_f1']:.4f}")
        print(f"  Std F1: {f1_comp['model2']['std_f1']:.4f}")
        
        comp = f1_comp['comparison']
        print(f"\nComparison:")
        print(f"  Mean F1 difference: {comp['mean_f1_diff']:.4f}")
        print(f"  Std F1 difference: {comp['std_f1_diff']:.4f}")
        print(f"  T-statistic: {comp['t_statistic']:.4f}")
        print(f"  P-value: {comp['p_value']:.4f}")
        print(f"  Significant (α=0.05): {'YES' if comp['significant'] else 'NO'}")
        
        print("\nPer-Class Results:")
        for cls_comp in f1_comp['per_class_comparison']:
            print(f"  {cls_comp['class_name']}:")
            print(f"    {results['model1_name']}: {cls_comp['f1_model1']:.4f}")
            print(f"    {results['model2_name']}: {cls_comp['f1_model2']:.4f}")
            print(f"    Difference: {cls_comp['difference']:+.4f} (Better: {cls_comp['better_model']})")
    
    # Paired t-test (if available)
    if 'paired_ttest' in results['tests']:
        print("\n" + "-"*80)
        print("Paired t-test (Episode-wise Accuracies)")
        print("-"*80)
        ttest = results['tests']['paired_ttest']
        print(f"T-statistic: {ttest['statistic']:.4f}")
        print(f"P-value: {ttest['p_value']:.4f}")
        print(f"Significant (α=0.05): {'YES' if ttest['significant'] else 'NO'}")
        print(f"Mean difference: {ttest['mean_diff']:.4f}")
        print(f"95% CI: [{ttest['confidence_interval_95']['lower']:.4f}, "
              f"{ttest['confidence_interval_95']['upper']:.4f}]")
        print(f"\nInterpretation: {ttest['interpretation']}")
    
    # Wilcoxon test (if available)
    if 'wilcoxon' in results['tests']:
        print("\n" + "-"*80)
        print("Wilcoxon Signed-Rank Test (Non-parametric)")
        print("-"*80)
        wilc = results['tests']['wilcoxon']
        print(f"Statistic: {wilc['statistic']:.4f}")
        print(f"P-value: {wilc['p_value']:.4f}")
        print(f"Significant (α=0.05): {'YES' if wilc['significant'] else 'NO'}")
        print(f"Median difference: {wilc['median_diff']:.4f}")
        print(f"\nInterpretation: {wilc['interpretation']}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    summary = results['summary']
    print(f"McNemar's test significant: {'YES' if summary['mcnemar_significant'] else 'NO'}")
    print(f"Per-class F1 significant: {'YES' if summary['per_class_f1_significant'] else 'NO'}")
    print(f"Overall significant difference: {'YES' if summary['overall_significant'] else 'NO'}")
    print(f"\n{summary['recommendation']}")
    print("="*80 + "\n")
