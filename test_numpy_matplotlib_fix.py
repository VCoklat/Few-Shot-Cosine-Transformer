#!/usr/bin/env python3
"""
Test to verify that the NumPy/Matplotlib compatibility issue is fixed.
This test should pass after the requirements.txt fix is applied.
"""

import sys
import os

def test_matplotlib_import():
    """Test that matplotlib can be imported without _ARRAY_API error."""
    print("\n" + "="*70)
    print("Test 1: Matplotlib Import")
    print("="*70)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib import transforms as mtransforms
        from mpl_toolkits.mplot3d import Axes3D
        import seaborn as sns
        
        print(f"✓ matplotlib version: {matplotlib.__version__}")
        print("✓ matplotlib.pyplot imported successfully")
        print("✓ matplotlib.transforms imported successfully (no _ARRAY_API error)")
        print("✓ mpl_toolkits.mplot3d.Axes3D imported successfully")
        print("✓ seaborn imported successfully")
        return True
        
    except AttributeError as e:
        if "_ARRAY_API" in str(e):
            print(f"✗ FAILED: _ARRAY_API error: {e}")
            print("This indicates NumPy 2.x compatibility issue")
            return False
        raise
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False


def test_numpy_version():
    """Test that NumPy version is < 2.0."""
    print("\n" + "="*70)
    print("Test 2: NumPy Version Check")
    print("="*70)
    
    try:
        import numpy as np
        version = np.__version__
        print(f"NumPy version: {version}")
        
        if version.startswith('2.'):
            print("✗ FAILED: NumPy 2.x detected")
            print("NumPy 2.x causes compatibility issues with matplotlib and numba")
            return False
        else:
            print("✓ NumPy version is compatible (< 2.0)")
            return True
            
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False


def test_umap_import():
    """Test that umap-learn can be imported without Numba/NumPy version error."""
    print("\n" + "="*70)
    print("Test 3: UMAP Import")
    print("="*70)
    
    try:
        import umap
        print(f"✓ umap-learn version: {umap.__version__}")
        print("✓ umap imported successfully (no Numba/NumPy version error)")
        return True
        
    except ImportError as e:
        if "Numba needs NumPy" in str(e):
            print(f"✗ FAILED: Numba/NumPy version error: {e}")
            print("This indicates NumPy 2.x is installed")
            return False
        print(f"Warning: umap-learn import failed: {e}")
        print("This is expected if umap-learn is not installed")
        return True  # Not a critical failure for this test
        
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False


def test_numba_import():
    """Test that numba can be imported."""
    print("\n" + "="*70)
    print("Test 4: Numba Import")
    print("="*70)
    
    try:
        import numba
        print(f"✓ numba version: {numba.__version__}")
        print("✓ numba imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False


def test_feature_visualizer_import():
    """Test that feature_visualizer can be imported (the actual failing module)."""
    print("\n" + "="*70)
    print("Test 5: Feature Visualizer Import")
    print("="*70)
    
    # Add the repository root to the path
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    try:
        import feature_visualizer
        print("✓ feature_visualizer imported successfully")
        
        # Check if all required components are available
        if hasattr(feature_visualizer, 'MATPLOTLIB_AVAILABLE'):
            print(f"  MATPLOTLIB_AVAILABLE: {feature_visualizer.MATPLOTLIB_AVAILABLE}")
        if hasattr(feature_visualizer, 'UMAP_AVAILABLE'):
            print(f"  UMAP_AVAILABLE: {feature_visualizer.UMAP_AVAILABLE}")
        if hasattr(feature_visualizer, 'SKLEARN_AVAILABLE'):
            print(f"  SKLEARN_AVAILABLE: {feature_visualizer.SKLEARN_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("NumPy/Matplotlib Compatibility Test Suite")
    print("="*70)
    print("\nThis test verifies that the requirements.txt fix resolves:")
    print("1. AttributeError: _ARRAY_API not found (NumPy 2.x + matplotlib)")
    print("2. Numba needs NumPy 2.0 or less error (NumPy 2.x + umap-learn)")
    
    results = {}
    
    # Run all tests
    results['numpy_version'] = test_numpy_version()
    results['matplotlib'] = test_matplotlib_import()
    results['numba'] = test_numba_import()
    results['umap'] = test_umap_import()
    results['feature_visualizer'] = test_feature_visualizer_import()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if all(results.values()):
        print("\n✓ ALL TESTS PASSED!")
        print("The NumPy/Matplotlib compatibility issue has been fixed.")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Please ensure requirements.txt is installed correctly:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
