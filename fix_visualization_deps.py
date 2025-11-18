#!/usr/bin/env python3
"""
Fix visualization dependencies in environments where matplotlib/numpy compatibility issues exist.

This script handles the common issue where matplotlib was pre-installed with numpy 2.x,
causing import errors when numpy 1.x is installed per requirements.txt.

Usage:
    python fix_visualization_deps.py
    
Or from within Python:
    import fix_visualization_deps
    fix_visualization_deps.fix_dependencies()
"""

import sys
import subprocess
import os


def run_command(cmd, description=""):
    """Run a shell command and return the result."""
    if description:
        print(f"\n{description}...")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Warning: Command failed with exit code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr[:500]}")
        
        return result.returncode == 0, result.stdout
    except Exception as e:
        print(f"Error running command: {e}")
        return False, ""


def check_numpy_version():
    """Check if numpy is installed and its version."""
    try:
        import numpy as np
        version = np.__version__
        print(f"✓ NumPy version: {version}")
        
        if version.startswith('2.'):
            print("⚠ WARNING: NumPy 2.x detected. This may cause compatibility issues.")
            print("  Recommendation: Install numpy<2.0.0 as specified in requirements.txt")
            return False, version
        
        return True, version
    except ImportError:
        print("✗ NumPy is not installed")
        return False, None


def check_matplotlib():
    """Check if matplotlib can be imported."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import transforms
        print(f"✓ matplotlib version: {matplotlib.__version__}")
        return True
    except (ImportError, AttributeError) as e:
        if "_ARRAY_API" in str(e) or "multiarray" in str(e):
            print(f"✗ matplotlib import failed with compatibility error: {e}")
            print("  This is typically caused by numpy/matplotlib binary incompatibility")
            return False
        elif isinstance(e, ImportError):
            print(f"✗ matplotlib is not installed")
            return False
        else:
            print(f"✗ matplotlib import failed: {e}")
            return False


def check_dependencies():
    """Check all visualization dependencies."""
    print("="*70)
    print("Checking Visualization Dependencies")
    print("="*70)
    
    issues = []
    
    # Check numpy
    numpy_ok, numpy_version = check_numpy_version()
    if not numpy_ok:
        issues.append("numpy")
    
    # Check matplotlib
    if not check_matplotlib():
        issues.append("matplotlib")
    
    # Check other optional dependencies
    try:
        import numba
        print(f"✓ numba version: {numba.__version__}")
    except ImportError:
        print("⚠ numba is not installed (optional for UMAP)")
        issues.append("numba")
    
    try:
        import umap
        print(f"✓ umap-learn version: {umap.__version__}")
    except ImportError as e:
        if "Numba needs NumPy" in str(e):
            print(f"✗ umap-learn import failed: {e}")
            issues.append("umap-learn")
        else:
            print(f"⚠ umap-learn is not installed (optional)")
    
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        print("✓ scikit-learn is installed")
    except ImportError:
        print("✗ scikit-learn is not installed")
        issues.append("scikit-learn")
    
    return issues


def fix_dependencies(force=False):
    """
    Fix visualization dependencies by reinstalling them.
    
    Args:
        force: If True, force reinstallation even if checks pass
    """
    print("\n" + "="*70)
    print("Fixing Visualization Dependencies")
    print("="*70)
    
    if not force:
        issues = check_dependencies()
        if not issues:
            print("\n✓ All dependencies are working correctly!")
            return True
        
        print(f"\n⚠ Found issues with: {', '.join(issues)}")
    
    print("\nThis script will:")
    print("  1. Ensure numpy<2.0.0 is installed")
    print("  2. Reinstall matplotlib and other visualization dependencies")
    print("  3. Verify all imports work correctly")
    
    response = input("\nProceed with fix? (y/n): ").lower().strip()
    if response != 'y':
        print("Fix cancelled.")
        return False
    
    # Step 1: Install/downgrade numpy
    print("\n" + "-"*70)
    print("Step 1: Installing numpy<2.0.0")
    print("-"*70)
    success, _ = run_command(
        f"{sys.executable} -m pip install --user 'numpy>=1.23.0,<2.0.0'",
        "Installing numpy"
    )
    
    if not success:
        print("✗ Failed to install numpy. Please check your pip configuration.")
        return False
    
    # Step 2: Reinstall matplotlib and visualization dependencies
    print("\n" + "-"*70)
    print("Step 2: Reinstalling visualization dependencies")
    print("-"*70)
    
    packages = [
        "'matplotlib>=3.5.0,<3.9.0'",
        "'numba>=0.57.0,<0.60.0'",
        "'scikit-learn>=1.0.0'",
        "'scipy>=1.9.0'",
        "seaborn",
        "pandas",
        "plotly",
        "'umap-learn>=0.5.3'"
    ]
    
    cmd = f"{sys.executable} -m pip install --user --force-reinstall --no-cache-dir {' '.join(packages)}"
    success, _ = run_command(cmd, "Reinstalling packages (this may take a few minutes)")
    
    if not success:
        print("✗ Failed to reinstall packages. Trying individual installation...")
        
        # Try installing packages one by one
        failed = []
        for package in packages:
            print(f"\nInstalling {package}...")
            cmd = f"{sys.executable} -m pip install --user --force-reinstall --no-cache-dir {package}"
            success, _ = run_command(cmd)
            if not success:
                failed.append(package)
        
        if failed:
            print(f"\n✗ Failed to install: {', '.join(failed)}")
            print("You may need to install these manually.")
    
    # Step 3: Verify installation
    print("\n" + "-"*70)
    print("Step 3: Verifying installation")
    print("-"*70)
    
    # Need to reload modules
    print("\nPlease restart your Python session and run:")
    print("  python -c 'import feature_visualizer; print(\"Success!\")'")
    print("\nOr run this script again to verify:")
    print("  python fix_visualization_deps.py --verify")
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix visualization dependencies for Few-Shot Cosine Transformer"
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only check dependencies without fixing'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reinstallation even if checks pass'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        issues = check_dependencies()
        if issues:
            print(f"\n✗ Found issues with: {', '.join(issues)}")
            print("Run without --verify to fix these issues.")
            sys.exit(1)
        else:
            print("\n✓ All dependencies are working correctly!")
            sys.exit(0)
    else:
        success = fix_dependencies(force=args.force)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
