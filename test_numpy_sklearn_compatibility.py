#!/usr/bin/env python3
"""
Test script to verify numpy and scikit-learn compatibility.

This script tests that the numpy/scikit-learn binary incompatibility issue
(ValueError: numpy.dtype size changed) has been resolved.

Run this after installing requirements.txt to verify the fix.
"""

import sys


def test_imports():
    """Test that all required imports work without errors."""
    print("Testing numpy/scikit-learn compatibility...")
    print("=" * 60)
    
    try:
        # Test basic numpy import
        import numpy as np
        print(f"✓ NumPy imported successfully (version {np.__version__})")
        
        # Test sklearn import
        import sklearn
        print(f"✓ Scikit-learn imported successfully (version {sklearn.__version__})")
        
        # Test the specific import that was failing
        import sklearn.metrics as metrics
        print("✓ sklearn.metrics imported successfully")
        
        # Test murmurhash import (this is where the error occurred)
        from sklearn.utils.murmurhash import murmurhash3_32
        print("✓ sklearn.utils.murmurhash imported successfully")
        
        # Test the specific functions used in train_test.py
        from sklearn.metrics import confusion_matrix, f1_score
        print("✓ confusion_matrix and f1_score imported successfully")
        
        # Test that functions actually work
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        cm = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"✓ Functions work correctly (F1 score: {f1:.4f})")
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("The numpy/scikit-learn compatibility issue is fixed.")
        return 0
        
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            print("\n❌ COMPATIBILITY ERROR DETECTED!")
            print(f"Error: {e}")
            print("\nThe numpy.dtype size changed error still occurs.")
            print("Try reinstalling with:")
            print('  pip install --upgrade --force-reinstall "numpy>=1.24.0,<2.0.0" "scikit-learn>=1.3.0,<1.5.0"')
            return 1
        else:
            raise
            
    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(test_imports())
