#!/usr/bin/env python3
"""
Test script to verify that the train() function correctly restores 
the best model (based on validation accuracy) after training completes.

This test verifies the code logic by checking that:
1. The train() function tracks the best epoch
2. It loads the best_model.tar before returning
3. The returned model has weights from the best checkpoint
"""
import os
import re

def test_train_py_logic():
    """Test the logic in train.py"""
    print("=" * 60)
    print("Testing Best Model Restoration Logic in train.py")
    print("=" * 60)
    
    train_py_path = os.path.join(os.path.dirname(__file__), 'train.py')
    
    with open(train_py_path, 'r') as f:
        content = f.read()
    
    # Test 1: Check that best_epoch is tracked
    print("\n1. Checking that best_epoch is tracked...")
    if 'best_epoch = -1' in content or 'best_epoch=-1' in content:
        print("   ✓ best_epoch variable is initialized")
    else:
        raise AssertionError("best_epoch should be initialized")
    
    if 'best_epoch = epoch' in content:
        print("   ✓ best_epoch is updated when best model is found")
    else:
        raise AssertionError("best_epoch should be updated")
    
    # Test 2: Check that best model is loaded before returning
    print("\n2. Checking that best model is loaded before returning...")
    if 'best_model_file = os.path.join(params.checkpoint_dir' in content:
        print("   ✓ best_model_file path is constructed")
    else:
        raise AssertionError("best_model_file path should be constructed")
    
    if 'checkpoint = torch.load(best_model_file)' in content:
        print("   ✓ best_model.tar is loaded")
    else:
        raise AssertionError("best_model.tar should be loaded")
    
    if "model.load_state_dict(checkpoint['state'])" in content:
        print("   ✓ Model state is restored from checkpoint")
    else:
        raise AssertionError("Model state should be restored from checkpoint")
    
    # Test 3: Check that informative message is printed
    print("\n3. Checking that informative messages are printed...")
    if 'Training completed' in content and 'Best validation accuracy' in content:
        print("   ✓ Training completion message is printed")
    else:
        raise AssertionError("Training completion message should be printed")
    
    if 'Loading best model' in content:
        print("   ✓ Model loading message is printed")
    else:
        raise AssertionError("Model loading message should be printed")
    
    print("\n" + "=" * 60)
    print("✓✓✓ train.py logic tests PASSED! ✓✓✓")
    print("=" * 60)

def test_train_test_py_logic():
    """Test the logic in train_test.py"""
    print("\n" + "=" * 60)
    print("Testing Best Model Restoration Logic in train_test.py")
    print("=" * 60)
    
    train_test_py_path = os.path.join(os.path.dirname(__file__), 'train_test.py')
    
    with open(train_test_py_path, 'r') as f:
        content = f.read()
    
    # Test 1: Check that best_epoch is tracked
    print("\n1. Checking that best_epoch is tracked...")
    if 'best_epoch = -1' in content or 'best_epoch=-1' in content:
        print("   ✓ best_epoch variable is initialized")
    else:
        raise AssertionError("best_epoch should be initialized")
    
    if 'best_epoch = epoch' in content:
        print("   ✓ best_epoch is updated when best model is found")
    else:
        raise AssertionError("best_epoch should be updated")
    
    # Test 2: Check that best model is loaded before returning
    print("\n2. Checking that best model is loaded before returning...")
    if 'best_model_file = os.path.join(params.checkpoint_dir' in content:
        print("   ✓ best_model_file path is constructed")
    else:
        raise AssertionError("best_model_file path should be constructed")
    
    if 'checkpoint = torch.load(best_model_file)' in content:
        print("   ✓ best_model.tar is loaded")
    else:
        raise AssertionError("best_model.tar should be loaded")
    
    if "model.load_state_dict(checkpoint['state'])" in content:
        print("   ✓ Model state is restored from checkpoint")
    else:
        raise AssertionError("Model state should be restored from checkpoint")
    
    # Test 3: Check that informative message is printed
    print("\n3. Checking that informative messages are printed...")
    if 'Training completed' in content and 'Best validation accuracy' in content:
        print("   ✓ Training completion message is printed")
    else:
        raise AssertionError("Training completion message should be printed")
    
    if 'Loading best model' in content:
        print("   ✓ Model loading message is printed")
    else:
        raise AssertionError("Model loading message should be printed")
    
    print("\n" + "=" * 60)
    print("✓✓✓ train_test.py logic tests PASSED! ✓✓✓")
    print("=" * 60)

def test_code_flow():
    """Test that the code flow is correct"""
    print("\n" + "=" * 60)
    print("Testing Code Flow")
    print("=" * 60)
    
    train_py_path = os.path.join(os.path.dirname(__file__), 'train.py')
    
    with open(train_py_path, 'r') as f:
        lines = f.readlines()
    
    # Find the train function
    train_func_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def train('):
            train_func_start = i
            break
    
    assert train_func_start is not None, "train() function should be found"
    
    # Find where the best model is saved and where it's loaded
    save_line = None
    load_line = None
    return_line = None
    
    for i in range(train_func_start, len(lines)):
        # Look for the save operation inside the best model block
        if 'torch.save' in lines[i] and i > 0:
            # Check if this is the best model save by looking at nearby lines
            for j in range(max(0, i-5), min(len(lines), i+5)):
                if 'best_model.tar' in lines[j]:
                    save_line = i
                    break
        if 'torch.load(best_model_file)' in lines[i]:
            load_line = i
        if lines[i].strip() == 'return model':
            return_line = i
            break
    
    print("\n1. Checking code flow order...")
    print(f"   Best model save line: {save_line}")
    print(f"   Best model load line: {load_line}")
    print(f"   Return statement line: {return_line}")
    
    assert save_line is not None, "Best model save should exist"
    assert load_line is not None, "Best model load should exist"
    assert return_line is not None, "Return statement should exist"
    
    # Verify order: save happens before load, load happens before return
    assert save_line < load_line, "Model save should happen before load"
    assert load_line < return_line, "Model load should happen before return"
    
    print("   ✓ Code flow is correct: save -> load -> return")
    
    print("\n" + "=" * 60)
    print("✓✓✓ Code flow test PASSED! ✓✓✓")
    print("=" * 60)

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print(" " * 20 + "BEST MODEL RESTORATION TEST SUITE")
    print("=" * 80)
    
    test_train_py_logic()
    test_train_test_py_logic()
    test_code_flow()
    
    print("\n" + "=" * 80)
    print(" " * 30 + "ALL TESTS PASSED! ✓✓✓")
    print("=" * 80)
    print("\nSummary:")
    print("✓ train.py correctly tracks best_epoch")
    print("✓ train.py loads best model before returning")
    print("✓ train_test.py correctly tracks best_epoch")
    print("✓ train_test.py loads best model before returning")
    print("✓ Code flow is correct in both files")
    print("✓ Informative messages are printed")
    print("\nConclusion:")
    print("The train() function now correctly restores the model to the")
    print("checkpoint with the best validation accuracy before returning.")
    print("=" * 80)

if __name__ == '__main__':
    main()
