#!/usr/bin/env python3
"""
Simple test to verify visualization imports work correctly.
This is a quick sanity check for users to run after fixing dependencies.
"""

import sys


def test_imports():
    """Test all visualization-related imports."""
    print("="*70)
    print("Testing Visualization Imports")
    print("="*70)
    
    all_ok = True
    
    # Test 1: NumPy
    print("\n1. Testing NumPy...")
    try:
        import numpy as np
        print(f"   ✓ NumPy {np.__version__}")
        
        if np.__version__.startswith('2.'):
            print(f"   ⚠ Warning: NumPy 2.x may cause compatibility issues")
            all_ok = False
    except ImportError as e:
        print(f"   ✗ Failed: {e}")
        all_ok = False
    
    # Test 2: Matplotlib
    print("\n2. Testing Matplotlib...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import transforms
        from mpl_toolkits.mplot3d import Axes3D
        print(f"   ✓ matplotlib {matplotlib.__version__}")
    except (ImportError, AttributeError) as e:
        print(f"   ✗ Failed: {e}")
        if "_ARRAY_API" in str(e) or "multiarray" in str(e):
            print(f"   → This is a numpy/matplotlib compatibility issue")
            print(f"   → Run: python fix_visualization_deps.py")
        all_ok = False
    
    # Test 3: Scikit-learn
    print("\n3. Testing Scikit-learn...")
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import sklearn
        print(f"   ✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"   ✗ Failed: {e}")
        all_ok = False
    
    # Test 4: Numba
    print("\n4. Testing Numba...")
    try:
        import numba
        print(f"   ✓ numba {numba.__version__}")
    except ImportError as e:
        print(f"   ✗ Failed: {e}")
        all_ok = False
    
    # Test 5: UMAP
    print("\n5. Testing UMAP...")
    try:
        import umap
        print(f"   ✓ umap-learn {umap.__version__}")
    except ImportError as e:
        print(f"   ⚠ Optional: {e}")
        if "Numba needs NumPy" in str(e):
            print(f"   → NumPy version incompatibility detected")
            print(f"   → Run: python fix_visualization_deps.py")
    
    # Test 6: Feature visualizer module
    print("\n6. Testing Feature Visualizer Module...")
    try:
        import feature_visualizer
        print(f"   ✓ feature_visualizer module imported")
        print(f"   - MATPLOTLIB_AVAILABLE: {feature_visualizer.MATPLOTLIB_AVAILABLE}")
        print(f"   - SKLEARN_AVAILABLE: {feature_visualizer.SKLEARN_AVAILABLE}")
        print(f"   - UMAP_AVAILABLE: {feature_visualizer.UMAP_AVAILABLE}")
        
        if not feature_visualizer.MATPLOTLIB_AVAILABLE:
            print(f"   ⚠ Matplotlib not available in feature_visualizer")
            all_ok = False
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_ok = False
    
    # Test 7: Other visualization dependencies
    print("\n7. Testing Other Dependencies...")
    try:
        import seaborn
        print(f"   ✓ seaborn {seaborn.__version__}")
    except ImportError:
        print(f"   ⚠ seaborn not installed (optional)")
    
    try:
        import pandas
        print(f"   ✓ pandas {pandas.__version__}")
    except ImportError:
        print(f"   ⚠ pandas not installed")
        all_ok = False
    
    try:
        import plotly
        print(f"   ✓ plotly {plotly.__version__}")
    except ImportError:
        print(f"   ⚠ plotly not installed (optional)")
    
    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("✓ ALL CRITICAL IMPORTS SUCCESSFUL!")
        print("\nYou can now use all visualization features.")
        print("Try: python example_visualization.py")
        return 0
    else:
        print("✗ SOME IMPORTS FAILED")
        print("\nTo fix dependency issues, run:")
        print("  python fix_visualization_deps.py")
        print("\nOr manually:")
        print("  pip install 'numpy>=1.23.0,<2.0.0'")
        print("  pip install --force-reinstall --no-cache-dir 'matplotlib>=3.5.0,<3.9.0'")
        return 1


if __name__ == '__main__':
    sys.exit(test_imports())
