"""
Test to simulate the exact error from the problem statement
"""

import json
import numpy as np
import sys
import os
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_analysis import comprehensive_feature_analysis


def test_exact_error_scenario():
    """
    Simulate the exact scenario from the error:
    - Create feature data with float32 dtype (as used by PyTorch)
    - Run comprehensive_feature_analysis
    - Try to save to JSON (the step that was failing)
    """
    print("=" * 80)
    print("Testing Exact Error Scenario from Problem Statement")
    print("=" * 80)
    print()
    
    # Create sample feature data with float32 (typical PyTorch output)
    np.random.seed(42)
    features = np.random.randn(600, 1600).astype(np.float32)
    labels = np.random.randint(0, 10, 600)
    
    print(f"Feature shape: {features.shape}")
    print(f"Feature dtype: {features.dtype}")
    print()
    
    # Run feature analysis (this generates the results)
    print("Running comprehensive_feature_analysis...")
    feature_analysis_results = {}
    feature_analysis_results['proposed'] = {
        'collapse': {'std_per_dimension': features.std(axis=0)},
        'utilization': {},
        'comprehensive': comprehensive_feature_analysis(features, labels)
    }
    print("✓ Analysis completed")
    print()
    
    # Simulate the code from run_experiments.py line 1160-1170
    import copy
    save_results = copy.deepcopy(feature_analysis_results)
    
    # Remove std_per_dimension for JSON serialization (too large)
    for model_name in save_results:
        if 'std_per_dimension' in save_results[model_name]['collapse']:
            del save_results[model_name]['collapse']['std_per_dimension']
    
    print("Attempting to save to JSON (this was failing before)...")
    
    # This is the exact line that was failing in the traceback
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(save_results, f, indent=2)
            temp_path = f.name
        
        print(f"✓ Successfully saved to JSON: {temp_path}")
        
        # Verify we can read it back
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        print("✓ Successfully loaded from JSON")
        
        # Clean up
        os.unlink(temp_path)
        
        print()
        print("=" * 80)
        print("SUCCESS: The exact error scenario is now fixed!")
        print("=" * 80)
        
        return True
        
    except TypeError as e:
        print(f"✗ FAILED with TypeError: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("FAILED: The error still exists")
        print("=" * 80)
        return False


if __name__ == '__main__':
    success = test_exact_error_scenario()
    sys.exit(0 if success else 1)
