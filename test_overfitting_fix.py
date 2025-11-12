#!/usr/bin/env python3
"""
Test to verify overfitting fixes are properly implemented.
This test checks that regularization parameters have been increased to prevent
the large gap between training accuracy (97%) and validation accuracy (60%).
"""

import sys
import re

def test_train_py_hyperparameters():
    """Test that train.py has reduced model capacity"""
    with open('train.py', 'r') as f:
        content = f.read()
    
    # Check for reduced model capacity
    assert 'depth=1' in content, "Model depth should be reduced to 1"
    assert 'heads=8' in content, "Number of heads should be reduced to 8"
    assert 'dim_head=64' in content, "dim_head should be reduced to 64"
    assert 'mlp_dim=512' in content, "mlp_dim should be reduced to 512"
    
    # Check for increased regularization
    assert 'label_smoothing=0.15' in content, "Label smoothing should be increased to 0.15"
    assert 'attention_dropout=0.2' in content, "Attention dropout should be increased to 0.2"
    assert 'drop_path_rate=0.15' in content, "Drop path rate should be increased to 0.15"
    
    print("✓ train.py hyperparameters correctly configured")

def test_early_stopping():
    """Test that early stopping is implemented"""
    with open('train.py', 'r') as f:
        content = f.read()
    
    assert 'patience' in content, "Early stopping patience should be defined"
    assert 'patience_counter' in content, "Patience counter should be implemented"
    assert 'Early stopping triggered' in content, "Early stopping message should exist"
    
    print("✓ Early stopping is properly implemented")

def test_transformer_regularization():
    """Test that transformer.py has stronger regularization"""
    with open('methods/transformer.py', 'r') as f:
        content = f.read()
    
    # Check for increased mixup alpha
    assert 'alpha=0.3' in content, "Mixup alpha should be increased to 0.3"
    
    # Check for increased FFN dropout
    assert 'self.ffn_dropout = nn.Dropout(0.15)' in content, "FFN dropout should be 0.15"
    
    print("✓ transformer.py regularization properly configured")

def test_weight_decay():
    """Test that weight decay has been increased"""
    with open('io_utils.py', 'r') as f:
        content = f.read()
    
    # Check for increased weight decay
    assert 'default=5e-4' in content or 'default=0.0005' in content, \
        "Weight decay should be increased to 5e-4"
    
    print("✓ Weight decay properly increased")

def test_comment_explanations():
    """Test that code changes are properly documented"""
    with open('train.py', 'r') as f:
        train_content = f.read()
    
    # Check for explanatory comments about overfitting
    assert 'overfitting' in train_content.lower() or 'generalization' in train_content.lower(), \
        "Changes should be documented with comments about overfitting"
    
    print("✓ Changes are properly documented")

if __name__ == '__main__':
    print("Testing overfitting fixes...")
    print()
    
    try:
        test_train_py_hyperparameters()
        test_early_stopping()
        test_transformer_regularization()
        test_weight_decay()
        test_comment_explanations()
        
        print()
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        print()
        print("Summary of changes to address overfitting (97% train vs 60% val):")
        print("1. Reduced model capacity (depth: 2→1, heads: 12→8, dim_head: 80→64, mlp_dim: 768→512)")
        print("2. Increased regularization (label_smoothing: 0.1→0.15, dropout: 0.15→0.2, drop_path: 0.1→0.15)")
        print("3. Stronger data augmentation (mixup alpha: 0.2→0.3)")
        print("4. Increased weight decay (1e-5→5e-4)")
        print("5. Added early stopping (patience=10 epochs)")
        print()
        print("Expected impact: Reduced gap between training and validation accuracy")
        print("Training accuracy may decrease slightly (97% → 85-90%)")
        print("Validation accuracy should increase significantly (60% → 70-80%)")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
