"""
Unit tests for the enhanced transformer components.
Tests Mahalanobis classifier, VIC regularization, and dynamic weight controller.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mahalanobis_classifier():
    """Test Mahalanobis distance classifier."""
    print("\n=== Testing MahalanobisClassifier ===")
    
    # Import module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location('mahalanobis_classifier', 
        'methods/mahalanobis_classifier.py')
    mahal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mahal_module)
    MahalanobisClassifier = mahal_module.MahalanobisClassifier
    
    # Create test data
    n_way = 5
    k_shot = 5
    n_query = 10
    d = 64
    
    classifier = MahalanobisClassifier(shrinkage_alpha=0.1)
    
    # Generate random support embeddings
    support_embeddings = torch.randn(n_way, k_shot, d)
    prototypes = support_embeddings.mean(dim=1)  # Simple mean prototypes
    query_embeddings = torch.randn(n_query, d)
    
    # Forward pass
    logits = classifier(query_embeddings, support_embeddings, prototypes)
    
    # Check output shape
    assert logits.shape == (n_query, n_way), f"Expected shape ({n_query}, {n_way}), got {logits.shape}"
    
    # Check that logits are finite
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values"
    
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print("✓ MahalanobisClassifier test passed!")
    
    return True


def test_vic_regularization():
    """Test VIC regularization."""
    print("\n=== Testing VICRegularization ===")
    
    # Import module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location('vic_regularization', 
        'methods/vic_regularization.py')
    vic_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vic_module)
    VICRegularization = vic_module.VICRegularization
    
    # Create test data
    n_way = 5
    k_shot = 5
    d = 64
    
    vic = VICRegularization(target_std=1.0, eps=1e-4)
    
    # Generate random support embeddings and prototypes
    support_embeddings = torch.randn(n_way, k_shot, d)
    prototypes = torch.randn(n_way, d)
    
    # Compute VIC stats
    v_loss, c_loss, stats = vic.compute_vic_stats(support_embeddings, prototypes)
    
    # Check outputs
    assert torch.isfinite(v_loss), "Variance loss is not finite"
    assert torch.isfinite(c_loss), "Covariance loss is not finite"
    assert v_loss >= 0, "Variance loss should be non-negative"
    assert c_loss >= 0, "Covariance loss should be non-negative"
    
    print(f"✓ Variance loss: {v_loss.item():.6f}")
    print(f"✓ Covariance loss: {c_loss.item():.6f}")
    print(f"✓ Mean std: {stats['mean_std']:.4f}")
    print(f"✓ Std range: [{stats['min_std']:.4f}, {stats['max_std']:.4f}]")
    print("✓ VICRegularization test passed!")
    
    return True


def test_dynamic_weight_controller():
    """Test dynamic weight controller."""
    print("\n=== Testing DynamicWeightController ===")
    
    # Import module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location('vic_regularization', 
        'methods/vic_regularization.py')
    vic_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vic_module)
    DynamicWeightController = vic_module.DynamicWeightController
    
    # Test uncertainty weighting
    print("\n--- Testing Uncertainty Weighting ---")
    controller = DynamicWeightController(
        n_losses=3, method='uncertainty', 
        init_weights=[9.0, 0.5, 0.5], bounds=(0.25, 4.0)
    )
    
    # Create dummy losses
    losses = [torch.tensor(2.0), torch.tensor(0.5), torch.tensor(0.3)]
    
    # Compute total loss
    total_loss = controller(losses)
    
    # Check output
    assert torch.isfinite(total_loss), "Total loss is not finite"
    assert total_loss > 0, "Total loss should be positive"
    
    # Get weights
    weights = controller.get_weights()
    print(f"✓ Learned weights: {weights.detach().numpy()}")
    print(f"✓ Total loss: {total_loss.item():.6f}")
    
    # Test weighted method
    print("\n--- Testing Fixed Weighting ---")
    controller2 = DynamicWeightController(
        n_losses=3, method='fixed', 
        init_weights=[9.0, 0.5, 0.5], bounds=(0.25, 4.0)
    )
    
    total_loss2 = controller2(losses)
    weights2 = controller2.get_weights()
    print(f"✓ Fixed weights: {weights2.detach().numpy()}")
    print(f"✓ Total loss: {total_loss2.item():.6f}")
    
    print("✓ DynamicWeightController test passed!")
    
    return True


def test_integration():
    """Test integration of all components."""
    print("\n=== Testing Integration ===")
    
    # Import modules directly
    import importlib.util
    
    spec = importlib.util.spec_from_file_location('mahalanobis_classifier', 
        'methods/mahalanobis_classifier.py')
    mahal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mahal_module)
    MahalanobisClassifier = mahal_module.MahalanobisClassifier
    
    spec = importlib.util.spec_from_file_location('vic_regularization', 
        'methods/vic_regularization.py')
    vic_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vic_module)
    VICRegularization = vic_module.VICRegularization
    DynamicWeightController = vic_module.DynamicWeightController
    
    # Create test scenario
    n_way = 5
    k_shot = 5
    n_query = 10
    d = 64
    
    # Initialize components
    classifier = MahalanobisClassifier()
    vic = VICRegularization()
    controller = DynamicWeightController(n_losses=3, method='uncertainty', 
                                        init_weights=[9.0, 0.5, 0.5])
    
    # Generate data
    support_embeddings = torch.randn(n_way, k_shot, d)
    prototypes = support_embeddings.mean(dim=1)
    query_embeddings = torch.randn(n_query, d)
    
    # 1. Compute classification logits (Invariance)
    logits = classifier(query_embeddings, support_embeddings, prototypes)
    targets = torch.randint(0, n_way, (n_query,))
    loss_I = torch.nn.functional.cross_entropy(logits, targets)
    
    # 2. Compute VIC regularization
    loss_V, loss_C, stats = vic.compute_vic_stats(support_embeddings, prototypes)
    
    # 3. Combine with dynamic weighting
    losses = [loss_I, loss_V, loss_C]
    total_loss = controller(losses)
    
    print(f"✓ Classification loss (I): {loss_I.item():.6f}")
    print(f"✓ Variance loss (V): {loss_V.item():.6f}")
    print(f"✓ Covariance loss (C): {loss_C.item():.6f}")
    print(f"✓ Total weighted loss: {total_loss.item():.6f}")
    
    # Test backward pass
    total_loss.backward()
    print("✓ Backward pass successful")
    
    print("✓ Integration test passed!")
    
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Running Enhanced Transformer Component Tests")
    print("=" * 60)
    
    try:
        test_mahalanobis_classifier()
        test_vic_regularization()
        test_dynamic_weight_controller()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
