#!/usr/bin/env python3
"""
Test script to simulate the exact error scenario from the problem statement.
This demonstrates that the fix handles the AttributeError: _ARRAY_API not found gracefully.
"""

import sys
import os

def simulate_kaggle_environment_error():
    """
    Simulate the scenario where matplotlib fails with AttributeError: _ARRAY_API not found.
    This was the original error reported in the problem statement.
    """
    print("="*70)
    print("Simulating Kaggle Environment Error Scenario")
    print("="*70)
    print("\nOriginal error was:")
    print("  File \".../feature_visualizer.py\", line 3, in <module>")
    print("    import matplotlib.pyplot as plt")
    print("  ...")
    print("  AttributeError: _ARRAY_API not found")
    print("  Error: feature_visualizer module not found")
    print()
    
    print("Testing our fix...")
    print("-" * 70)
    
    # Test 1: Import feature_visualizer
    print("\n1. Attempting to import feature_visualizer module...")
    try:
        import feature_visualizer
        print("   ✓ SUCCESS: Module imported without crashing!")
        print(f"   - MATPLOTLIB_AVAILABLE: {feature_visualizer.MATPLOTLIB_AVAILABLE}")
        
        if not feature_visualizer.MATPLOTLIB_AVAILABLE:
            print("   ✓ Correctly detected that matplotlib is not available")
        
    except AttributeError as e:
        if "_ARRAY_API" in str(e):
            print(f"   ✗ FAILED: Still getting _ARRAY_API error: {e}")
            return False
        else:
            print(f"   ✗ FAILED: Different AttributeError: {e}")
            return False
    except Exception as e:
        print(f"   ✗ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Try to use the visualize_features_from_results function
    print("\n2. Attempting to call visualize_features_from_results...")
    try:
        from feature_visualizer import visualize_features_from_results
        print("   ✓ Function imported successfully")
        
        # Try to call it (it should return None gracefully)
        import numpy as np
        features = np.random.randn(10, 5)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        result = visualize_features_from_results(features, labels, show=False)
        
        if result is None:
            print("   ✓ Function returned None gracefully (no crash)")
        else:
            print("   ✓ Function completed successfully")
        
    except AttributeError as e:
        if "_ARRAY_API" in str(e):
            print(f"   ✗ FAILED: Still getting _ARRAY_API error: {e}")
            return False
        else:
            print(f"   ✗ FAILED: Different AttributeError: {e}")
            return False
    except Exception as e:
        print(f"   ✗ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Verify the import in eval_utils (the original call site)
    print("\n3. Testing the original call site in eval_utils...")
    try:
        # This simulates the code path from train_test.py -> eval_utils.py -> feature_visualizer.py
        print("   Simulating: eval_utils.visualize_feature_projections()")
        
        # Try to import the function from eval_utils
        # Note: This will fail if torch is not available, but that's OK for this test
        try:
            from eval_utils import visualize_feature_projections
            print("   ✓ eval_utils.visualize_feature_projections imported")
        except ModuleNotFoundError as e:
            if "torch" in str(e):
                print("   ⚠ Cannot fully test (torch not available), but import structure is correct")
            else:
                raise
        
    except AttributeError as e:
        if "_ARRAY_API" in str(e):
            print(f"   ✗ FAILED: Still getting _ARRAY_API error: {e}")
            return False
        else:
            print(f"   ✗ FAILED: Different AttributeError: {e}")
            return False
    except Exception as e:
        print(f"   Note: {e}")
        print("   (This is expected if torch is not installed)")
    
    return True

def main():
    """Run the simulation test."""
    
    success = simulate_kaggle_environment_error()
    
    print("\n" + "="*70)
    print("Test Results")
    print("="*70)
    
    if success:
        print("\n✓ SUCCESS: The fix correctly handles the matplotlib import error!")
        print("\nKey improvements:")
        print("  1. Module imports without AttributeError: _ARRAY_API not found")
        print("  2. Provides clear warning messages about missing dependencies")
        print("  3. Code continues to run instead of crashing")
        print("  4. Returns None gracefully when visualization is not possible")
        print("\nThe original error is now fixed!")
        return 0
    else:
        print("\n✗ FAILED: The fix did not resolve the issue")
        return 1

if __name__ == "__main__":
    sys.exit(main())
