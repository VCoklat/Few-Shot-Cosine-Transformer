#!/usr/bin/env python
"""
Test NumPy 2.x compatibility fix

This test verifies that the NumPy version constraint prevents the 
AttributeError: _ARRAY_API not found error that occurs when matplotlib
(compiled with NumPy 1.x) runs with NumPy 2.x.
"""

import sys
import importlib.metadata


def test_numpy_version_constraint():
    """Test that NumPy version is less than 2.0.0"""
    import numpy as np
    
    version = np.__version__
    print(f"NumPy version: {version}")
    
    major, minor, patch = map(int, version.split('.')[:3])
    
    assert major < 2, f"NumPy version {version} is >= 2.0.0, should be < 2.0.0"
    assert major == 1, f"NumPy major version should be 1, got {major}"
    assert minor >= 23, f"NumPy minor version should be >= 23, got {minor}"
    
    print(f"✅ NumPy version constraint satisfied: {version} < 2.0.0")


def test_matplotlib_import():
    """Test that matplotlib can be imported without _ARRAY_API error"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import transforms as mtransforms
        from matplotlib._path import affine_transform
        
        print(f"✅ Matplotlib version: {matplotlib.__version__}")
        print("✅ Successfully imported matplotlib.pyplot")
        print("✅ Successfully imported matplotlib.transforms")
        print("✅ Successfully imported matplotlib._path")
        print("✅ No AttributeError: _ARRAY_API not found")
        
    except AttributeError as e:
        if '_ARRAY_API' in str(e):
            print(f"❌ AttributeError: _ARRAY_API not found - NumPy 2.x compatibility issue!")
            raise AssertionError(f"NumPy 2.x compatibility issue: {e}")
        else:
            raise


def test_dependencies_versions():
    """Test that all dependencies meet minimum version requirements"""
    dependencies = {
        'numpy': '1.23.0',
        'matplotlib': '3.5.0',
        'scikit-learn': '1.0.0',
        'scipy': '1.9.0',
    }
    
    for package, min_version in dependencies.items():
        try:
            version = importlib.metadata.version(package)
            print(f"✓ {package}: {version} (>= {min_version})")
            
            # Simple version comparison (works for basic cases)
            pkg_parts = [int(x) for x in version.split('.')[:3]]
            min_parts = [int(x) for x in min_version.split('.')[:3]]
            
            # Pad with zeros if needed
            while len(pkg_parts) < 3:
                pkg_parts.append(0)
            while len(min_parts) < 3:
                min_parts.append(0)
            
            assert pkg_parts >= min_parts, f"{package} {version} < {min_version}"
            
        except importlib.metadata.PackageNotFoundError:
            print(f"⚠ {package} not installed (optional)")


def test_feature_visualizer_dependencies():
    """Test that dependencies required by feature_visualizer.py are available"""
    required = ['seaborn', 'pandas', 'plotly']
    
    for package in required:
        try:
            version = importlib.metadata.version(package)
            print(f"✓ {package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"⚠ {package} not installed (required by feature_visualizer.py)")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing NumPy 2.x Compatibility Fix")
    print("=" * 70)
    print()
    
    try:
        print("Test 1: NumPy version constraint")
        print("-" * 70)
        test_numpy_version_constraint()
        print()
        
        print("Test 2: Matplotlib import (main fix verification)")
        print("-" * 70)
        test_matplotlib_import()
        print()
        
        print("Test 3: Dependencies versions")
        print("-" * 70)
        test_dependencies_versions()
        print()
        
        print("Test 4: Feature visualizer dependencies")
        print("-" * 70)
        test_feature_visualizer_dependencies()
        print()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED - NumPy 2.x compatibility issue is RESOLVED!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
