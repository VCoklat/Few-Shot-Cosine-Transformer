"""
Unit tests for compare_models_mcnemar.py

This test suite verifies the core functionality of the McNemar comparison script.
Note: This test requires the full PyTorch environment to be installed.
For a lighter-weight test that doesn't require PyTorch, see test_mcnemar_simple.py
which reimplements the core McNemar's test logic for verification purposes.
"""

import os
import sys
import unittest
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions we want to test
from eval_utils import compare_models_mcnemar
from ablation_study import mcnemar_test, compute_contingency_table


class TestMcNemarComparison(unittest.TestCase):
    """Test cases for McNemar's test comparison functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample predictions and true labels
        np.random.seed(42)
        self.n_samples = 1000
        self.true_labels = np.random.randint(0, 5, self.n_samples)
        
        # Model A: 70% accuracy
        self.predictions_a = self.true_labels.copy()
        errors_a = np.random.choice(self.n_samples, int(self.n_samples * 0.3), replace=False)
        self.predictions_a[errors_a] = (self.predictions_a[errors_a] + 1) % 5
        
        # Model B: 65% accuracy (slightly worse)
        self.predictions_b = self.true_labels.copy()
        errors_b = np.random.choice(self.n_samples, int(self.n_samples * 0.35), replace=False)
        self.predictions_b[errors_b] = (self.predictions_b[errors_b] + 1) % 5
    
    def test_contingency_table(self):
        """Test contingency table computation"""
        n00, n01, n10, n11 = compute_contingency_table(
            self.predictions_a, self.predictions_b, self.true_labels
        )
        
        # Verify all samples are accounted for
        self.assertEqual(n00 + n01 + n10 + n11, self.n_samples)
        
        # Verify values are non-negative
        self.assertGreaterEqual(n00, 0)
        self.assertGreaterEqual(n01, 0)
        self.assertGreaterEqual(n10, 0)
        self.assertGreaterEqual(n11, 0)
        
        print(f"Contingency table: n00={n00}, n01={n01}, n10={n10}, n11={n11}")
    
    def test_mcnemar_test_basic(self):
        """Test basic McNemar's test functionality"""
        result = mcnemar_test(
            self.predictions_a, self.predictions_b, self.true_labels
        )
        
        # Check that all expected keys are present
        expected_keys = [
            'contingency_table', 'statistic', 'p_value',
            'significant_at_0.05', 'significant_at_0.01',
            'algorithm_a_better', 'algorithm_b_better',
            'discordant_pairs', 'effect_description', 'test_type'
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        
        # Check value types
        self.assertIsInstance(result['p_value'], float)
        self.assertIsInstance(result['statistic'], float)
        self.assertIsInstance(result['significant_at_0.05'], bool)
        self.assertIsInstance(result['discordant_pairs'], int)
        
        # P-value should be between 0 and 1
        self.assertGreaterEqual(result['p_value'], 0.0)
        self.assertLessEqual(result['p_value'], 1.0)
        
        print(f"McNemar test result: p={result['p_value']:.4f}, "
              f"statistic={result['statistic']:.4f}, "
              f"significant={result['significant_at_0.05']}")
    
    def test_compare_models_mcnemar(self):
        """Test the high-level comparison function"""
        result = compare_models_mcnemar(
            self.predictions_a,
            self.predictions_b,
            self.true_labels,
            model_a_name="Model A",
            model_b_name="Model B"
        )
        
        # Check that model names are set
        self.assertEqual(result['model_a_name'], "Model A")
        self.assertEqual(result['model_b_name'], "Model B")
        
        # Check that accuracies are computed
        self.assertIn('model_a_accuracy', result)
        self.assertIn('model_b_accuracy', result)
        
        # Verify accuracy calculations
        expected_acc_a = np.mean(self.predictions_a == self.true_labels)
        expected_acc_b = np.mean(self.predictions_b == self.true_labels)
        
        self.assertAlmostEqual(result['model_a_accuracy'], expected_acc_a, places=4)
        self.assertAlmostEqual(result['model_b_accuracy'], expected_acc_b, places=4)
        
        print(f"Model A accuracy: {result['model_a_accuracy']:.4f}")
        print(f"Model B accuracy: {result['model_b_accuracy']:.4f}")
    
    def test_identical_predictions(self):
        """Test McNemar's test with identical predictions"""
        # When predictions are identical, should have no discordant pairs
        result = mcnemar_test(
            self.predictions_a, self.predictions_a, self.true_labels
        )
        
        self.assertEqual(result['discordant_pairs'], 0)
        self.assertEqual(result['p_value'], 1.0)
        self.assertFalse(result['significant_at_0.05'])
        self.assertIn("No difference", result['effect_description'])
        
        print("Identical predictions test passed")
    
    def test_clearly_different_models(self):
        """Test with models that are clearly different"""
        # Model A: 90% accuracy
        predictions_good = self.true_labels.copy()
        errors_good = np.random.choice(self.n_samples, int(self.n_samples * 0.1), replace=False)
        predictions_good[errors_good] = (predictions_good[errors_good] + 1) % 5
        
        # Model B: 50% accuracy (much worse)
        predictions_bad = self.true_labels.copy()
        errors_bad = np.random.choice(self.n_samples, int(self.n_samples * 0.5), replace=False)
        predictions_bad[errors_bad] = (predictions_bad[errors_bad] + 1) % 5
        
        result = mcnemar_test(predictions_good, predictions_bad, self.true_labels)
        
        # Should be highly significant
        self.assertTrue(result['significant_at_0.05'])
        self.assertTrue(result['significant_at_0.01'])
        self.assertTrue(result['algorithm_a_better'])
        self.assertFalse(result['algorithm_b_better'])
        
        print(f"Clearly different models: p={result['p_value']:.6f}")
    
    def test_exact_test_with_few_discordant_pairs(self):
        """Test that exact binomial test is used with few discordant pairs"""
        # Create predictions with only a few discordant pairs
        predictions_a = np.zeros(100, dtype=int)
        predictions_b = np.zeros(100, dtype=int)
        true_labels = np.zeros(100, dtype=int)
        
        # Create exactly 10 discordant pairs
        predictions_a[0:5] = 1  # A wrong, B correct
        predictions_b[5:10] = 1  # A correct, B wrong
        
        result = mcnemar_test(predictions_a, predictions_b, true_labels)
        
        # Should use exact test
        self.assertIn('exact', result['test_type'].lower())
        
        print(f"Exact test used: {result['test_type']}")


def run_tests():
    """Run all tests"""
    print("="*80)
    print("Running McNemar Comparison Tests")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMcNemarComparison)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
