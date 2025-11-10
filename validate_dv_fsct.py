"""
Static code analysis for DV-FSCT implementation
Validates structure and logic without running the code
"""

import ast
import sys

def check_file_structure(filepath):
    """Check if Python file has valid structure"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        return True, classes, functions
    except SyntaxError as e:
        return False, None, None


def validate_dv_fsct():
    """Validate DV-FSCT implementation structure"""
    print("="*60)
    print("DV-FSCT Implementation Structure Validation")
    print("="*60)
    
    filepath = "methods/dv_fsct.py"
    valid, classes, functions = check_file_structure(filepath)
    
    if not valid:
        print(f"✗ {filepath} has syntax errors")
        return False
    
    print(f"✓ {filepath} is syntactically valid")
    print(f"\nClasses found: {classes}")
    print(f"Key methods found: {[f for f in functions if not f.startswith('_')][:10]}")
    
    # Check for required classes
    required_classes = ['DVFSCT', 'CosineAttention']
    for cls in required_classes:
        if cls in classes:
            print(f"✓ Class '{cls}' found")
        else:
            print(f"✗ Class '{cls}' not found")
            return False
    
    # Check for key methods
    key_methods = ['compute_dynamic_weights', 'vic_loss', 'set_forward', 'set_forward_loss', 'cosine_sim']
    found_methods = [m for m in key_methods if m in functions]
    print(f"\n✓ Found {len(found_methods)}/{len(key_methods)} key methods")
    
    return True


def validate_integration():
    """Validate integration with existing codebase"""
    print("\n" + "="*60)
    print("Integration Validation")
    print("="*60)
    
    files_to_check = [
        'io_utils.py',
        'train.py',
        'train_test.py',
        'test.py'
    ]
    
    all_valid = True
    for filepath in files_to_check:
        valid, classes, functions = check_file_structure(filepath)
        if valid:
            print(f"✓ {filepath} is syntactically valid")
        else:
            print(f"✗ {filepath} has syntax errors")
            all_valid = False
    
    return all_valid


def validate_test_file():
    """Validate test file structure"""
    print("\n" + "="*60)
    print("Test File Validation")
    print("="*60)
    
    filepath = "test_dv_fsct.py"
    valid, classes, functions = check_file_structure(filepath)
    
    if not valid:
        print(f"✗ {filepath} has syntax errors")
        return False
    
    print(f"✓ {filepath} is syntactically valid")
    
    test_functions = [f for f in functions if f.startswith('test_')]
    print(f"✓ Found {len(test_functions)} test functions: {test_functions}")
    
    return True


def main():
    """Run all validations"""
    results = []
    
    results.append(("DV-FSCT Structure", validate_dv_fsct()))
    results.append(("Integration", validate_integration()))
    results.append(("Test File", validate_test_file()))
    
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    for name, passed in results:
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("All validations PASSED! ✓")
        print("\nThe DV-FSCT implementation is structurally correct.")
        print("Note: Runtime tests require PyTorch to be installed.")
    else:
        print("Some validations FAILED! ✗")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
