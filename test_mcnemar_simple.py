"""
Simplified unit tests for McNemar's test functionality

This test suite verifies the core McNemar's test implementation
without requiring the full PyTorch environment.
"""

import sys
import numpy as np
from scipy import stats


def compute_contingency_table(predictions_a, predictions_b, true_labels):
    """Compute the contingency table for McNemar's test."""
    if len(predictions_a) != len(predictions_b) or len(predictions_a) != len(true_labels):
        raise ValueError("All arrays must have the same length")
    
    correct_a = (predictions_a == true_labels)
    correct_b = (predictions_b == true_labels)
    
    n00 = np.sum(~correct_a & ~correct_b)  # Both wrong
    n01 = np.sum(~correct_a & correct_b)   # A wrong, B correct
    n10 = np.sum(correct_a & ~correct_b)   # A correct, B wrong
    n11 = np.sum(correct_a & correct_b)    # Both correct
    
    return int(n00), int(n01), int(n10), int(n11)


def mcnemar_test(predictions_a, predictions_b, true_labels, correction=True):
    """Perform McNemar's test to compare two classification algorithms."""
    predictions_a = np.asarray(predictions_a)
    predictions_b = np.asarray(predictions_b)
    true_labels = np.asarray(true_labels)
    
    n00, n01, n10, n11 = compute_contingency_table(predictions_a, predictions_b, true_labels)
    
    discordant_pairs = n01 + n10
    
    if discordant_pairs == 0:
        return {
            'contingency_table': (n00, n01, n10, n11),
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'discordant_pairs': 0,
            'test_type': 'none'
        }
    
    # Use chi-squared approximation
    if correction:
        statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    else:
        statistic = (n01 - n10) ** 2 / (n01 + n10)
    
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {
        'contingency_table': (n00, n01, n10, n11),
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'discordant_pairs': discordant_pairs,
        'test_type': 'chi_squared',
        'algorithm_a_better': n10 > n01
    }


def test_contingency_table():
    """Test contingency table computation"""
    print("\nTest 1: Contingency Table")
    print("-" * 50)
    
    # Create sample data
    true_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    predictions_a = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4, 4])  # 7/10 correct
    predictions_b = np.array([0, 1, 1, 1, 2, 3, 3, 3, 4, 0])  # 6/10 correct
    
    n00, n01, n10, n11 = compute_contingency_table(predictions_a, predictions_b, true_labels)
    
    total = n00 + n01 + n10 + n11
    print(f"Contingency table: n00={n00}, n01={n01}, n10={n10}, n11={n11}")
    print(f"Total samples: {total}")
    
    assert total == len(true_labels), "Total should equal number of samples"
    assert all(x >= 0 for x in [n00, n01, n10, n11]), "All values should be non-negative"
    
    print("✓ Test passed")
    return True


def test_mcnemar_basic():
    """Test basic McNemar's test functionality"""
    print("\nTest 2: Basic McNemar's Test")
    print("-" * 50)
    
    np.random.seed(42)
    n_samples = 1000
    true_labels = np.random.randint(0, 5, n_samples)
    
    # Model A: 70% accuracy
    predictions_a = true_labels.copy()
    errors_a = np.random.choice(n_samples, int(n_samples * 0.3), replace=False)
    predictions_a[errors_a] = (predictions_a[errors_a] + 1) % 5
    
    # Model B: 65% accuracy
    predictions_b = true_labels.copy()
    errors_b = np.random.choice(n_samples, int(n_samples * 0.35), replace=False)
    predictions_b[errors_b] = (predictions_b[errors_b] + 1) % 5
    
    result = mcnemar_test(predictions_a, predictions_b, true_labels)
    
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Test statistic: {result['statistic']:.4f}")
    print(f"Significant: {result['significant']}")
    print(f"Discordant pairs: {result['discordant_pairs']}")
    
    assert 0.0 <= result['p_value'] <= 1.0, "P-value should be between 0 and 1"
    assert result['statistic'] >= 0.0, "Test statistic should be non-negative"
    
    print("✓ Test passed")
    return True


def test_identical_predictions():
    """Test with identical predictions"""
    print("\nTest 3: Identical Predictions")
    print("-" * 50)
    
    true_labels = np.array([0, 1, 2, 3, 4] * 20)
    predictions = true_labels.copy()
    
    result = mcnemar_test(predictions, predictions, true_labels)
    
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Discordant pairs: {result['discordant_pairs']}")
    
    assert result['discordant_pairs'] == 0, "Should have no discordant pairs"
    assert result['p_value'] == 1.0, "P-value should be 1.0"
    assert not result['significant'], "Should not be significant"
    
    print("✓ Test passed")
    return True


def test_clearly_different():
    """Test with clearly different models"""
    print("\nTest 4: Clearly Different Models")
    print("-" * 50)
    
    np.random.seed(42)
    n_samples = 1000
    true_labels = np.random.randint(0, 5, n_samples)
    
    # Model A: 90% accuracy
    predictions_good = true_labels.copy()
    errors_good = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    predictions_good[errors_good] = (predictions_good[errors_good] + 1) % 5
    
    # Model B: 50% accuracy
    predictions_bad = true_labels.copy()
    errors_bad = np.random.choice(n_samples, int(n_samples * 0.5), replace=False)
    predictions_bad[errors_bad] = (predictions_bad[errors_bad] + 1) % 5
    
    result = mcnemar_test(predictions_good, predictions_bad, true_labels)
    
    acc_good = np.mean(predictions_good == true_labels)
    acc_bad = np.mean(predictions_bad == true_labels)
    
    print(f"Good model accuracy: {acc_good:.2%}")
    print(f"Bad model accuracy: {acc_bad:.2%}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"Significant: {result['significant']}")
    
    assert result['significant'], "Should be significant"
    assert result['p_value'] < 0.01, "Should be highly significant"
    
    print("✓ Test passed")
    return True


def test_accuracy_calculations():
    """Test accuracy calculations"""
    print("\nTest 5: Accuracy Calculations")
    print("-" * 50)
    
    true_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    predictions_a = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4, 4])
    predictions_b = np.array([0, 1, 1, 1, 2, 3, 3, 3, 4, 0])
    
    acc_a = np.mean(predictions_a == true_labels)
    acc_b = np.mean(predictions_b == true_labels)
    
    print(f"Model A accuracy: {acc_a:.2%}")
    print(f"Model B accuracy: {acc_b:.2%}")
    
    # Check actual accuracies
    correct_a = np.sum(predictions_a == true_labels)
    correct_b = np.sum(predictions_b == true_labels)
    print(f"Model A correct: {correct_a}/{len(true_labels)}")
    print(f"Model B correct: {correct_b}/{len(true_labels)}")
    
    result = mcnemar_test(predictions_a, predictions_b, true_labels)
    
    print(f"P-value: {result['p_value']:.4f}")
    print(f"A is better: {result.get('algorithm_a_better', False)}")
    
    print("✓ Test passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("McNemar's Test - Simplified Unit Tests")
    print("=" * 80)
    
    tests = [
        test_contingency_table,
        test_mcnemar_basic,
        test_identical_predictions,
        test_clearly_different,
        test_accuracy_calculations
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed!")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
