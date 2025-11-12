#!/usr/bin/env python3
"""
Verify that the validate_model function is properly defined in train_test.py
This is a simple syntax and import check without requiring full dependencies.
"""
import ast
import sys

def check_function_exists():
    """Parse train_test.py and check if validate_model function is defined"""
    print("Checking train_test.py for validate_model function...")
    
    with open('train_test.py', 'r') as f:
        content = f.read()
    
    # Parse the Python file
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"✗ Syntax error in train_test.py: {e}")
        return False
    
    # Find all function definitions
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    # Check if validate_model exists
    if 'validate_model' in functions:
        print("✓ validate_model function is defined")
        
        # Find the function node to check its parameters
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'validate_model':
                # Get parameter names
                params = [arg.arg for arg in node.args.args]
                print(f"  Parameters: {params}")
                
                # Check expected parameters
                if params == ['val_loader', 'model']:
                    print("✓ Function has correct parameters (val_loader, model)")
                else:
                    print(f"⚠ Warning: Expected parameters ['val_loader', 'model'], got {params}")
                
                # Check if function has a return statement
                has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                if has_return:
                    print("✓ Function has a return statement")
                else:
                    print("✗ Function is missing a return statement")
                    return False
                
                break
        
        # Check that validate_model is called in the train function
        validate_model_called = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'validate_model':
                    validate_model_called = True
                    break
        
        if validate_model_called:
            print("✓ validate_model is called in the code")
        else:
            print("⚠ Warning: validate_model might not be called")
        
        return True
    else:
        print("✗ validate_model function is NOT defined")
        print(f"Available functions: {', '.join(functions[:10])}...")
        return False

def main():
    print("=" * 60)
    print("VALIDATE_MODEL FUNCTION VERIFICATION")
    print("=" * 60)
    
    try:
        success = check_function_exists()
        
        if success:
            print("\n" + "=" * 60)
            print("✓ VERIFICATION PASSED")
            print("=" * 60)
            print("\nThe validate_model function is properly defined and")
            print("should resolve the NameError: name 'validate_model' is not defined")
            return 0
        else:
            print("\n" + "=" * 60)
            print("✗ VERIFICATION FAILED")
            print("=" * 60)
            return 1
            
    except FileNotFoundError:
        print("✗ Error: train_test.py not found in current directory")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
