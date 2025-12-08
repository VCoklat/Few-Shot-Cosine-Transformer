"""
Test to verify that McNemar's test results are JSON serializable
"""

import json
import numpy as np
import sys

# Import the mcnemar_test function from ablation_study
from ablation_study import mcnemar_test


def test_json_serialization_basic():
    """Test that mcnemar_test results can be serialized to JSON"""
    print("\nTest 1: Basic JSON Serialization")
    print("-" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    true_labels = np.random.randint(0, 5, n_samples)
    
    # Model A: 70% accuracy
    predictions_a = true_labels.copy()
    errors_a = np.random.choice(n_samples, int(n_samples * 0.3), replace=False)
    predictions_a[errors_a] = (predictions_a[errors_a] + 1) % 5
    
    # Model B: 65% accuracy
    predictions_b = true_labels.copy()
    errors_b = np.random.choice(n_samples, int(n_samples * 0.35), replace=False)
    predictions_b[errors_b] = (predictions_b[errors_b] + 1) % 5
    
    # Run McNemar's test
    result = mcnemar_test(predictions_a, predictions_b, true_labels)
    
    # Try to serialize to JSON
    try:
        json_str = json.dumps(result, indent=2)
        print("✓ Successfully serialized to JSON")
        
        # Verify we can deserialize it back
        deserialized = json.loads(json_str)
        print("✓ Successfully deserialized from JSON")
        
        # Check that key fields are present and have correct types
        assert isinstance(deserialized['p_value'], float), "p_value should be float"
        assert isinstance(deserialized['significant_at_0.05'], bool), "significant_at_0.05 should be bool"
        assert isinstance(deserialized['significant_at_0.01'], bool), "significant_at_0.01 should be bool"
        assert isinstance(deserialized['algorithm_a_better'], bool), "algorithm_a_better should be bool"
        assert isinstance(deserialized['algorithm_b_better'], bool), "algorithm_b_better should be bool"
        print("✓ All boolean fields are native Python bool type")
        
        return True
    except TypeError as e:
        print(f"✗ Failed to serialize to JSON: {e}")
        return False


def test_json_serialization_edge_cases():
    """Test JSON serialization with edge cases"""
    print("\nTest 2: Edge Case - Identical Predictions")
    print("-" * 50)
    
    # Test with identical predictions (no discordant pairs)
    true_labels = np.array([0, 1, 2, 3, 4] * 20)
    predictions = true_labels.copy()
    
    result = mcnemar_test(predictions, predictions, true_labels)
    
    try:
        json_str = json.dumps(result, indent=2)
        print("✓ Successfully serialized edge case to JSON")
        
        deserialized = json.loads(json_str)
        assert deserialized['discordant_pairs'] == 0
        assert isinstance(deserialized['significant_at_0.05'], bool)
        print("✓ Edge case handled correctly")
        
        return True
    except TypeError as e:
        print(f"✗ Failed to serialize edge case: {e}")
        return False


def test_json_serialization_with_file():
    """Test writing to an actual JSON file"""
    print("\nTest 3: Writing to JSON File")
    print("-" * 50)
    
    # Create sample data
    np.random.seed(123)
    n_samples = 200
    true_labels = np.random.randint(0, 5, n_samples)
    
    predictions_a = true_labels.copy()
    errors_a = np.random.choice(n_samples, int(n_samples * 0.25), replace=False)
    predictions_a[errors_a] = (predictions_a[errors_a] + 1) % 5
    
    predictions_b = true_labels.copy()
    errors_b = np.random.choice(n_samples, int(n_samples * 0.3), replace=False)
    predictions_b[errors_b] = (predictions_b[errors_b] + 1) % 5
    
    result = mcnemar_test(predictions_a, predictions_b, true_labels)
    
    try:
        # Write to file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(result, f, indent=2)
            temp_path = f.name
        
        print(f"✓ Successfully wrote to file: {temp_path}")
        
        # Read back and verify
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        print("✓ Successfully read from file")
        
        # Verify types
        assert isinstance(loaded['significant_at_0.05'], bool)
        assert isinstance(loaded['significant_at_0.01'], bool)
        print("✓ All types preserved correctly")
        
        # Clean up
        os.unlink(temp_path)
        
        return True
    except Exception as e:
        print(f"✗ Failed file operations: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("McNemar's Test JSON Serialization Tests")
    print("=" * 80)
    
    tests = [
        test_json_serialization_basic,
        test_json_serialization_edge_cases,
        test_json_serialization_with_file
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
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
