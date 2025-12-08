"""
Test to verify compute_confidence_interval returns JSON-serializable types
"""

import json
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_analysis import compute_confidence_interval


def test_confidence_interval_serialization():
    """Test that compute_confidence_interval returns native Python types"""
    print("=" * 80)
    print("Testing compute_confidence_interval JSON Serialization")
    print("=" * 80)
    print()
    
    # Create sample accuracy data with float32
    np.random.seed(42)
    accuracies = np.random.uniform(0.7, 0.9, 100).astype(np.float32)
    
    print(f"Accuracies dtype: {accuracies.dtype}")
    print(f"Sample values: {accuracies[:5]}")
    print()
    
    # Compute confidence interval
    mean, lower, upper = compute_confidence_interval(accuracies)
    
    print(f"Mean: {mean} (type: {type(mean).__name__})")
    print(f"Lower: {lower} (type: {type(lower).__name__})")
    print(f"Upper: {upper} (type: {type(upper).__name__})")
    print()
    
    # Verify types are native Python floats
    assert isinstance(mean, float) and not isinstance(mean, np.floating), \
        f"Mean should be Python float, got {type(mean)}"
    assert isinstance(lower, float) and not isinstance(lower, np.floating), \
        f"Lower should be Python float, got {type(lower)}"
    assert isinstance(upper, float) and not isinstance(upper, np.floating), \
        f"Upper should be Python float, got {type(upper)}"
    
    print("✓ All return values are native Python float types")
    print()
    
    # Test JSON serialization
    result = {
        'mean': mean,
        'confidence_interval': [lower, upper],
        'margin': (upper - lower) / 2
    }
    
    try:
        json_str = json.dumps(result, indent=2)
        print("✓ Successfully serialized to JSON")
        
        # Verify we can deserialize
        loaded = json.loads(json_str)
        print("✓ Successfully deserialized from JSON")
        
        print()
        print("=" * 80)
        print("SUCCESS: compute_confidence_interval is JSON-serializable!")
        print("=" * 80)
        
        return True
        
    except TypeError as e:
        print(f"✗ FAILED with TypeError: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("FAILED: compute_confidence_interval is not JSON-serializable")
        print("=" * 80)
        return False


if __name__ == '__main__':
    success = test_confidence_interval_serialization()
    sys.exit(0 if success else 1)
