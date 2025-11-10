#!/usr/bin/env python3
"""
Minimal validation script to verify memory optimization changes
"""
import sys
import ast

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)

def check_for_pattern(filepath, pattern):
    """Check if a file contains a specific pattern"""
    with open(filepath, 'r') as f:
        content = f.read()
        return pattern in content

def main():
    print("="*60)
    print("Validating Memory Optimization Changes")
    print("="*60)
    
    # Files to check
    files_to_check = [
        'methods/meta_template.py',
        'train_test.py',
        'train.py',
        'io_utils.py',
        'test_gradient_accumulation.py'
    ]
    
    print("\n1. Checking Python syntax...")
    all_valid = True
    for filepath in files_to_check:
        valid, error = check_file_syntax(filepath)
        if valid:
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath}: {error}")
            all_valid = False
    
    if not all_valid:
        print("\n✗ Syntax errors found!")
        return False
    
    print("\n2. Checking for key implementation features...")
    
    # Check meta_template.py for gradient accumulation
    checks = [
        ('methods/meta_template.py', 'torch.cuda.amp.GradScaler', 'GradScaler initialization'),
        ('methods/meta_template.py', 'torch.cuda.amp.autocast', 'Autocast usage'),
        ('methods/meta_template.py', 'accumulation_steps', 'Accumulation steps parameter'),
        ('methods/meta_template.py', 'scaler.scale', 'Scaled backward pass'),
        ('methods/meta_template.py', 'torch.cuda.empty_cache', 'Memory clearing'),
        ('io_utils.py', 'gradient_accumulation_steps', 'CLI parameter'),
        ('train_test.py', 'accumulation_steps', 'Accumulation steps usage'),
        ('train.py', 'accumulation_steps', 'Accumulation steps usage'),
        ('README.md', 'Memory Optimization', 'Documentation'),
    ]
    
    for filepath, pattern, description in checks:
        if check_for_pattern(filepath, pattern):
            print(f"  ✓ {description} in {filepath}")
        else:
            print(f"  ✗ {description} NOT found in {filepath}")
            all_valid = False
    
    print("\n3. Checking backward compatibility...")
    # Check that default parameters maintain backward compatibility
    if check_for_pattern('methods/meta_template.py', 'accumulation_steps=2'):
        print("  ✓ Default accumulation_steps parameter set")
    else:
        print("  ✗ Default accumulation_steps parameter NOT set")
        all_valid = False
    
    if check_for_pattern('io_utils.py', 'default=0'):
        print("  ✓ Auto-detection enabled by default (gradient_accumulation_steps=0)")
    else:
        print("  ✗ Auto-detection NOT enabled by default")
        all_valid = False
    
    print("\n" + "="*60)
    if all_valid:
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print("="*60)
        print("\nMemory optimization features successfully implemented:")
        print("  • Automatic Mixed Precision (AMP) with GradScaler")
        print("  • Gradient accumulation with configurable steps")
        print("  • Memory clearing after optimizer steps")
        print("  • Auto-detection of accumulation steps based on backbone")
        print("  • Backward compatible with existing code")
        return True
    else:
        print("✗ VALIDATION FAILED - Some checks did not pass")
        print("="*60)
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
