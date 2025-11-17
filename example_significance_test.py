#!/usr/bin/env python3
"""
Example script demonstrating how to use significance testing and per-class F1 score
comparison for few-shot learning models.

This script shows:
1. How to compute and display per-class F1 scores for a single model
2. How to compare two models using statistical significance tests
3. How to integrate significance testing into the evaluation pipeline

Usage:
    # Compare two models
    python example_significance_test.py --model1 FSCT_cosine --model2 CTX_cosine --dataset miniImagenet

    # Show per-class F1 scores for a single model
    python example_significance_test.py --model1 FSCT_cosine --dataset miniImagenet --single-model
"""

import numpy as np
import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from significance_test import (
    compute_per_class_f1,
    compare_per_class_f1,
    comprehensive_significance_test,
    pretty_print_significance_test
)


def example_single_model_evaluation():
    """
    Example: Evaluate a single model and show per-class F1 scores.
    
    This demonstrates how to use compute_per_class_f1() to analyze
    performance across different classes in few-shot learning.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Per-Class F1 Score Analysis for a Single Model")
    print("="*80 + "\n")
    
    # Simulate evaluation results from a few-shot learning model
    # In real usage, these would come from model.set_forward() on test episodes
    np.random.seed(42)
    
    n_episodes = 1000
    n_way = 5  # 5-way classification
    n_query = 15  # 15 query samples per class per episode
    
    # Simulate predictions (each episode has n_way * n_query predictions)
    all_true = []
    all_pred = []
    
    print("Simulating 1000 episodes of 5-way 5-shot classification...")
    print("(In real usage, these would be actual model predictions)\n")
    
    for episode in range(n_episodes):
        # True labels for this episode (0, 1, 2, 3, 4)
        y_true = np.repeat(np.arange(n_way), n_query)
        
        # Simulate predictions with varying accuracy per class
        y_pred = y_true.copy()
        for cls in range(n_way):
            cls_mask = y_true == cls
            # Different accuracy for each class: 90%, 85%, 80%, 75%, 70%
            error_rate = 0.10 + cls * 0.05
            n_errors = int(error_rate * np.sum(cls_mask))
            if n_errors > 0:
                error_indices = np.random.choice(np.where(cls_mask)[0], 
                                                 size=n_errors, replace=False)
                y_pred[error_indices] = (y_pred[error_indices] + 1) % n_way
        
        all_true.append(y_true)
        all_pred.append(y_pred)
    
    # Concatenate all episodes
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    
    # Compute per-class F1 scores
    f1_results = compute_per_class_f1(y_true, y_pred, n_way)
    
    # Display results
    print("="*80)
    print("PER-CLASS F1 SCORE RESULTS")
    print("="*80)
    print(f"\n📊 Overall Metrics:")
    print(f"  Macro F1:    {f1_results['macro_f1']:.4f}")
    print(f"  Micro F1:    {f1_results['micro_f1']:.4f}")
    print(f"  Weighted F1: {f1_results['weighted_f1']:.4f}")
    print(f"  Std F1:      {f1_results['std_f1']:.4f}")
    print(f"  Range:       [{f1_results['min_f1']:.4f}, {f1_results['max_f1']:.4f}]")
    
    print(f"\n📋 Per-Class Breakdown:")
    class_names = [f"Way {i}" for i in range(n_way)]
    for i, (f1, prec, rec) in enumerate(zip(f1_results['per_class_f1'],
                                              f1_results['per_class_precision'],
                                              f1_results['per_class_recall'])):
        print(f"  {class_names[i]}:")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
    
    print("\n💡 Interpretation:")
    best_class = np.argmax(f1_results['per_class_f1'])
    worst_class = np.argmin(f1_results['per_class_f1'])
    print(f"  Best performing class:  {class_names[best_class]} "
          f"(F1={f1_results['per_class_f1'][best_class]:.4f})")
    print(f"  Worst performing class: {class_names[worst_class]} "
          f"(F1={f1_results['per_class_f1'][worst_class]:.4f})")
    
    if f1_results['std_f1'] > 0.05:
        print(f"  ⚠️  High variance in F1 scores (std={f1_results['std_f1']:.4f}) "
              "suggests imbalanced performance across classes.")
    else:
        print(f"  ✓ Low variance in F1 scores (std={f1_results['std_f1']:.4f}) "
              "suggests balanced performance across classes.")
    
    print("\n" + "="*80 + "\n")


def example_two_model_comparison():
    """
    Example: Compare two models using comprehensive significance testing.
    
    This demonstrates how to use comprehensive_significance_test() to
    statistically compare two few-shot learning models.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Statistical Comparison of Two Models")
    print("="*80 + "\n")
    
    # Simulate evaluation results from two models
    np.random.seed(42)
    
    n_episodes = 1000
    n_way = 5
    n_query = 15
    
    # Simulate predictions for Model 1 (better model)
    all_true = []
    all_pred1 = []
    all_pred2 = []
    episode_acc1 = []
    episode_acc2 = []
    
    print("Simulating 1000 episodes for two models...")
    print("Model 1: Few-Shot Cosine Transformer")
    print("Model 2: Cross Transformer\n")
    
    for episode in range(n_episodes):
        y_true = np.repeat(np.arange(n_way), n_query)
        
        # Model 1: Better model (80% avg accuracy)
        y_pred1 = y_true.copy()
        n_wrong1 = int(0.20 * len(y_true))
        wrong_idx1 = np.random.choice(len(y_true), size=n_wrong1, replace=False)
        y_pred1[wrong_idx1] = (y_pred1[wrong_idx1] + 1) % n_way
        
        # Model 2: Worse model (75% avg accuracy)
        y_pred2 = y_true.copy()
        n_wrong2 = int(0.25 * len(y_true))
        wrong_idx2 = np.random.choice(len(y_true), size=n_wrong2, replace=False)
        y_pred2[wrong_idx2] = (y_pred2[wrong_idx2] + 1) % n_way
        
        all_true.append(y_true)
        all_pred1.append(y_pred1)
        all_pred2.append(y_pred2)
        
        episode_acc1.append(np.mean(y_pred1 == y_true))
        episode_acc2.append(np.mean(y_pred2 == y_true))
    
    # Concatenate all episodes
    y_true = np.concatenate(all_true)
    y_pred1 = np.concatenate(all_pred1)
    y_pred2 = np.concatenate(all_pred2)
    episode_acc1 = np.array(episode_acc1)
    episode_acc2 = np.array(episode_acc2)
    
    # Compute overall accuracies
    acc1 = np.mean(y_pred1 == y_true)
    acc2 = np.mean(y_pred2 == y_true)
    
    print(f"Model 1 Overall Accuracy: {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"Model 2 Overall Accuracy: {acc2:.4f} ({acc2*100:.2f}%)")
    print(f"Raw Difference: {acc1 - acc2:.4f} ({(acc1-acc2)*100:.2f}%)\n")
    
    # Run comprehensive significance test
    class_names = [f"Way {i}" for i in range(n_way)]
    results = comprehensive_significance_test(
        y_true, y_pred1, y_pred2,
        episode_acc1, episode_acc2,
        class_names,
        model1_name="Few-Shot Cosine Transformer",
        model2_name="Cross Transformer"
    )
    
    # Pretty print the results
    pretty_print_significance_test(results)
    
    # Additional insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if results['summary']['overall_significant']:
        print("\n✅ Statistical Significance Detected!")
        print("\nThe difference between the two models IS statistically significant.")
        print("This means the observed performance difference is unlikely due to chance.")
        print("\nRecommendation: The better performing model (Model 1) can be")
        print("confidently selected for production use.")
    else:
        print("\n❌ No Statistical Significance Detected")
        print("\nThe difference between the two models is NOT statistically significant.")
        print("This means the observed performance difference could be due to chance.")
        print("\nRecommendation: Consider:")
        print("  • Collecting more test episodes for higher statistical power")
        print("  • Looking at other factors (inference time, memory usage, etc.)")
        print("  • Using both models in an ensemble")
    
    print("\n" + "="*80 + "\n")


def example_integration_with_eval():
    """
    Example: Show how to integrate significance testing into eval_utils.
    
    This demonstrates the pattern for integrating significance tests
    into the existing evaluation pipeline.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Integration with Evaluation Pipeline")
    print("="*80 + "\n")
    
    print("To integrate significance testing into the evaluation pipeline,")
    print("add the following to your test.py or train_test.py:\n")
    
    code_example = """
# After evaluating your model with eval_utils.evaluate()
from significance_test import compute_per_class_f1

# Get predictions and labels from evaluation
y_true = all_true_labels  # Concatenated from all episodes
y_pred = all_predictions  # Concatenated from all episodes
n_way = params.n_way

# Compute per-class F1 scores
f1_results = compute_per_class_f1(y_true, y_pred, n_way)

# Display per-class results
print("\\n" + "="*80)
print("PER-CLASS F1 SCORES")
print("="*80)
print(f"Macro F1: {f1_results['macro_f1']:.4f}")
print(f"Micro F1: {f1_results['micro_f1']:.4f}")
print(f"Std F1: {f1_results['std_f1']:.4f}")

for i, f1 in enumerate(f1_results['per_class_f1']):
    print(f"  Class {i}: F1={f1:.4f}")

# To compare two models:
from significance_test import comprehensive_significance_test

results = comprehensive_significance_test(
    y_true, y_pred_model1, y_pred_model2,
    episode_acc1, episode_acc2,
    class_names=[f"Class {i}" for i in range(n_way)],
    model1_name="FSCT_cosine",
    model2_name="CTX_cosine"
)

# Print comparison results
from significance_test import pretty_print_significance_test
pretty_print_significance_test(results)
"""
    
    print(code_example)
    print("\n" + "="*80 + "\n")


def main():
    """Run all examples."""
    parser = argparse.ArgumentParser(
        description="Demonstrate significance testing for few-shot learning models"
    )
    parser.add_argument(
        '--example',
        type=str,
        choices=['single', 'compare', 'integration', 'all'],
        default='all',
        help='Which example to run (default: all)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SIGNIFICANCE TESTING EXAMPLES FOR FEW-SHOT LEARNING")
    print("="*80)
    print("\nThis script demonstrates how to use the new significance testing")
    print("features to analyze and compare few-shot learning models.\n")
    
    if args.example in ['single', 'all']:
        example_single_model_evaluation()
    
    if args.example in ['compare', 'all']:
        example_two_model_comparison()
    
    if args.example in ['integration', 'all']:
        example_integration_with_eval()
    
    print("="*80)
    print("✅ EXAMPLES COMPLETED")
    print("="*80)
    print("\nFor more information, see:")
    print("  • significance_test.py - Full implementation")
    print("  • test_significance.py - Unit tests")
    print("  • eval_utils.py - Integration with evaluation pipeline")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
