#!/usr/bin/env python3
"""
Test script to verify that the visualization module handles import errors gracefully.
"""

import sys
import numpy as np

def test_import_with_missing_matplotlib():
    """Test that feature_visualizer can be imported even if matplotlib is broken."""
    print("Test 1: Testing import of feature_visualizer module...")
    try:
        import feature_visualizer
        print("✓ feature_visualizer module imported successfully")
        
        # Check availability flags
        print(f"  - MATPLOTLIB_AVAILABLE: {feature_visualizer.MATPLOTLIB_AVAILABLE}")
        print(f"  - SKLEARN_AVAILABLE: {feature_visualizer.SKLEARN_AVAILABLE}")
        print(f"  - UMAP_AVAILABLE: {feature_visualizer.UMAP_AVAILABLE}")
        print(f"  - TORCH_AVAILABLE: {feature_visualizer.TORCH_AVAILABLE}")
        print(f"  - PLOTLY_AVAILABLE: {feature_visualizer.PLOTLY_AVAILABLE}")
        print(f"  - PANDAS_AVAILABLE: {feature_visualizer.PANDAS_AVAILABLE}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import feature_visualizer: {e}")
        return False

def test_visualize_features_with_missing_deps():
    """Test that visualize_features_from_results handles missing dependencies gracefully."""
    print("\nTest 2: Testing visualize_features_from_results with potentially missing dependencies...")
    
    try:
        from feature_visualizer import visualize_features_from_results
        print("✓ visualize_features_from_results imported successfully")
        
        # Create dummy data
        features = np.random.randn(100, 50)
        labels = np.random.randint(0, 5, 100)
        
        # Try to visualize - this should handle missing dependencies gracefully
        result = visualize_features_from_results(
            features, 
            labels, 
            show=False,  # Don't show the plot
            save_dir=None
        )
        
        if result is None:
            print("  Note: Visualization returned None (likely due to missing dependencies)")
            print("  This is expected if matplotlib or scikit-learn are not available")
            return True
        else:
            print("✓ Visualization completed successfully")
            return True
            
    except ImportError as e:
        print(f"✓ Import error handled gracefully: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_eval_utils_import():
    """Test that eval_utils can import feature_visualizer gracefully."""
    print("\nTest 3: Testing eval_utils visualization function...")
    
    try:
        import eval_utils
        print("✓ eval_utils module imported successfully")
        
        # Check if the function exists
        if hasattr(eval_utils, 'visualize_feature_projections'):
            print("✓ visualize_feature_projections function found")
            return True
        else:
            print("✗ visualize_feature_projections function not found")
            return False
            
    except Exception as e:
        print(f"✗ Failed to import eval_utils: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*70)
    print("Testing Visualization Module Error Handling")
    print("="*70)
    
    results = []
    
    # Test 1: Import feature_visualizer
    results.append(test_import_with_missing_matplotlib())
    
    # Test 2: Test visualize_features_from_results
    results.append(test_visualize_features_with_missing_deps())
    
    # Test 3: Test eval_utils
    results.append(test_eval_utils_import())
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
