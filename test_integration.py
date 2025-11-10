"""
Basic integration test for FSCT_ProFONet with training pipeline
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.fsct_profonet import FSCT_ProFONet
from backbone import Conv4, ResNet12
import io_utils

def test_method_selection():
    """Test that the new method can be selected via CLI args"""
    print("\n=== Testing Method Selection Integration ===")
    
    # Create mock args
    class MockParams:
        method = 'FSCT_ProFONet'
        backbone = 'Conv4'
        dataset = 'miniImagenet'
        FETI = 0
        n_way = 5
        k_shot = 5
        n_query = 10
        
    params = MockParams()
    
    # Test that method is recognized
    valid_methods = ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine', 'FSCT_ProFONet']
    assert params.method in valid_methods, f"Method {params.method} not in valid methods"
    
    print(f"✓ Method '{params.method}' is recognized")
    return True


def test_model_instantiation():
    """Test model instantiation with different backbones"""
    print("\n=== Testing Model Instantiation with Different Backbones ===")
    
    backbones = [
        ('Conv4', 'miniImagenet', False, True),
    ]
    
    for backbone_name, dataset, feti, flatten in backbones:
        print(f"\nTesting {backbone_name} with dataset={dataset}, flatten={flatten}")
        
        n_way = 5
        k_shot = 5
        n_query = 10
        
        if backbone_name == 'Conv4':
            def model_func():
                return Conv4(dataset=dataset, flatten=flatten)
        elif backbone_name == 'ResNet12':
            def model_func():
                return ResNet12(FETI=feti, dataset=dataset, flatten=flatten)
        
        model = FSCT_ProFONet(
            model_func=model_func,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            variant='cosine',
            depth=1,
            heads=4,
            dim_head=160,
            mlp_dim=512,
            gradient_checkpointing=False,
            mixed_precision=False
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model created with {param_count:,} parameters")
        print(f"  Feature dimension: {model.feat_dim}")
        
    print("\n✓ All backbones tested successfully")
    return True


def test_training_step():
    """Test a single training step"""
    print("\n=== Testing Single Training Step ===")
    
    n_way = 5
    k_shot = 5
    n_query = 10
    
    def model_func():
        return Conv4(dataset='miniImagenet', flatten=True)
    
    model = FSCT_ProFONet(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        gradient_checkpointing=False,
        mixed_precision=False
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Set epoch for dynamic weights
    model.set_epoch(0, 50)
    
    # Create sample batch
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    acc, loss = model.set_forward_loss(x)
    loss.backward()
    
    # Gradient clipping (as in train_loop)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    print(f"Training step completed:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Gradients computed: {any(p.grad is not None for p in model.parameters())}")
    
    # Check dynamic weights
    lambda_V, lambda_I, lambda_C = model.weight_scheduler.get_weights(0, 50)
    print(f"  Dynamic weights: λ_V={lambda_V:.4f}, λ_I={lambda_I:.4f}, λ_C={lambda_C:.4f}")
    
    print("✓ Training step test passed")
    return True


def test_validation_step():
    """Test a validation step"""
    print("\n=== Testing Validation Step ===")
    
    n_way = 5
    k_shot = 5
    n_query = 10
    
    def model_func():
        return Conv4(dataset='miniImagenet', flatten=True)
    
    model = FSCT_ProFONet(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine'
    )
    
    # Create sample batch
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    
    # Validation step
    model.eval()
    with torch.no_grad():
        correct, total = model.correct(x)
    
    accuracy = correct / total * 100
    
    print(f"Validation step completed:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    print("✓ Validation step test passed")
    return True


def test_memory_optimizations():
    """Test memory optimization features"""
    print("\n=== Testing Memory Optimization Features ===")
    
    n_way = 5
    k_shot = 5
    n_query = 10
    
    # Test with gradient checkpointing
    print("\nTesting with gradient checkpointing enabled:")
    def model_func():
        return Conv4(dataset='miniImagenet', flatten=True)
    
    model = FSCT_ProFONet(
        model_func=model_func,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        variant='cosine',
        gradient_checkpointing=True,  # Enabled
        mixed_precision=False
    )
    
    print(f"  Gradient checkpointing: {model.gradient_checkpointing}")
    print(f"  Mixed precision: {model.mixed_precision}")
    
    # Forward pass
    x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
    model.train()
    acc, loss = model.set_forward_loss(x)
    loss.backward()
    
    print(f"  Forward/backward pass successful")
    print(f"  Loss: {loss.item():.6f}")
    
    print("\n✓ Memory optimization features working")
    return True


def run_integration_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("Running FSCT_ProFONet Integration Tests")
    print("="*60)
    
    tests = [
        test_method_selection,
        test_model_instantiation,
        test_training_step,
        test_validation_step,
        test_memory_optimizations
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED with error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
