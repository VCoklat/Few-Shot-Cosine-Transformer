"""
Test to verify that feature analysis results are JSON serializable
"""

import json
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_analysis import (
    detect_feature_collapse,
    compute_feature_utilization,
    compute_diversity_score,
    analyze_feature_redundancy,
    compute_intraclass_consistency,
    identify_confusing_pairs,
    compute_imbalance_ratio,
    comprehensive_feature_analysis
)


def test_detect_feature_collapse():
    """Test that detect_feature_collapse results are JSON serializable"""
    print("\nTest 1: detect_feature_collapse")
    print("-" * 50)
    
    # Create sample data
    np.random.seed(42)
    features = np.random.randn(100, 50).astype(np.float32)
    
    result = detect_feature_collapse(features)
    
    try:
        json_str = json.dumps(result, indent=2)
        print("✓ Successfully serialized to JSON")
        
        # Verify we can deserialize it back
        deserialized = json.loads(json_str)
        print("✓ Successfully deserialized from JSON")
        
        # Check types
        assert isinstance(deserialized['collapsed_dimensions'], int)
        assert isinstance(deserialized['total_dimensions'], int)
        assert isinstance(deserialized['collapse_ratio'], float)
        print("✓ All types are native Python types")
        
        return True
    except TypeError as e:
        print(f"✗ Failed to serialize to JSON: {e}")
        return False


def test_compute_diversity_score():
    """Test that compute_diversity_score results are JSON serializable"""
    print("\nTest 2: compute_diversity_score")
    print("-" * 50)
    
    # Create sample data
    np.random.seed(42)
    features = np.random.randn(100, 50).astype(np.float32)
    labels = np.random.randint(0, 5, 100)
    
    result = compute_diversity_score(features, labels)
    
    try:
        json_str = json.dumps(result, indent=2)
        print("✓ Successfully serialized to JSON")
        
        deserialized = json.loads(json_str)
        print("✓ Successfully deserialized from JSON")
        
        # Check types
        assert isinstance(deserialized['mean_diversity'], float)
        assert isinstance(deserialized['per_class_diversity'], list)
        if deserialized['per_class_diversity']:
            assert isinstance(deserialized['per_class_diversity'][0], float)
        print("✓ All types are native Python types")
        
        return True
    except TypeError as e:
        print(f"✗ Failed to serialize to JSON: {e}")
        return False


def test_compute_intraclass_consistency():
    """Test that compute_intraclass_consistency results are JSON serializable"""
    print("\nTest 3: compute_intraclass_consistency")
    print("-" * 50)
    
    # Create sample data
    np.random.seed(42)
    features = np.random.randn(100, 50).astype(np.float32)
    labels = np.random.randint(0, 5, 100)
    
    result = compute_intraclass_consistency(features, labels)
    
    try:
        json_str = json.dumps(result, indent=2)
        print("✓ Successfully serialized to JSON")
        
        deserialized = json.loads(json_str)
        print("✓ Successfully deserialized from JSON")
        
        # Check types
        assert isinstance(deserialized['mean_euclidean_consistency'], float)
        assert isinstance(deserialized['per_class_euclidean'], list)
        print("✓ All types are native Python types")
        
        return True
    except TypeError as e:
        print(f"✗ Failed to serialize to JSON: {e}")
        return False


def test_comprehensive_feature_analysis():
    """Test that comprehensive_feature_analysis results are JSON serializable"""
    print("\nTest 4: comprehensive_feature_analysis")
    print("-" * 50)
    
    # Create sample data
    np.random.seed(42)
    features = np.random.randn(100, 50).astype(np.float32)
    labels = np.random.randint(0, 5, 100)
    
    result = comprehensive_feature_analysis(features, labels)
    
    try:
        json_str = json.dumps(result, indent=2)
        print("✓ Successfully serialized to JSON")
        
        deserialized = json.loads(json_str)
        print("✓ Successfully deserialized from JSON")
        
        # Check that nested structures are properly converted
        assert 'feature_collapse' in deserialized
        assert 'diversity_score' in deserialized
        assert 'intraclass_consistency' in deserialized
        print("✓ All nested structures are properly converted")
        
        return True
    except TypeError as e:
        print(f"✗ Failed to serialize to JSON: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_file_write():
    """Test writing to an actual JSON file"""
    print("\nTest 5: Writing to JSON File")
    print("-" * 50)
    
    # Create sample data
    np.random.seed(123)
    features = np.random.randn(200, 100).astype(np.float32)
    labels = np.random.randint(0, 10, 200)
    
    result = comprehensive_feature_analysis(features, labels)
    
    try:
        # Write to file
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(result, f, indent=2)
            temp_path = f.name
        
        print(f"✓ Successfully wrote to file: {temp_path}")
        
        # Read back and verify
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        print("✓ Successfully read from file")
        
        # Clean up
        os.unlink(temp_path)
        
        return True
    except Exception as e:
        print(f"✗ Failed file operations: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("Feature Analysis JSON Serialization Tests")
    print("=" * 80)
    
    tests = [
        test_detect_feature_collapse,
        test_compute_diversity_score,
        test_compute_intraclass_consistency,
        test_comprehensive_feature_analysis,
        test_json_file_write
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
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
