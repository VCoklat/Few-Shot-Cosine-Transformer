#!/usr/bin/env python
"""
Test script to verify the memory optimization for Mahalanobis classifier.
This simulates the training scenario with ResNet34 features.
"""

import torch
import torch.nn as nn
import sys
import gc

sys.path.insert(0, '.')

from methods.enhanced_transformer import EnhancedFewShotTransformer
from methods.mahalanobis_classifier import MahalanobisClassifier

def get_memory_allocated():
    """Get current GPU memory allocated in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

def test_forward_pass():
    """Test forward pass with high-dimensional features"""
    print("=" * 60)
    print("Testing Forward Pass with High-Dimensional Features")
    print("=" * 60)
    
    # Use CPU for testing if CUDA not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Simulate ResNet34 feature extractor
    class ResNet34FeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.final_feat_dim = 512 * 7 * 7  # 25088 - same as ResNet34
            
        def forward(self, x):
            batch_size = x.size(0)
            # Simulate feature extraction
            return torch.randn(batch_size, self.final_feat_dim, device=x.device)
    
    # Create model with dimensionality reduction
    print("\n1. Creating model with dim reduction (reduced_dim=512)...")
    model = EnhancedFewShotTransformer(
        model_func=ResNet34FeatureModel,
        n_way=5,
        k_shot=5,
        n_query=8,
        variant='cosine',
        reduced_dim=512,
        use_mahalanobis=True,
        use_vic=True,
        use_checkpoint=False
    ).to(device)
    
    print(f"   - Original feature dimension: {model.feat_dim}")
    print(f"   - Has dim_reduction: {model.dim_reduction is not None}")
    if model.dim_reduction is not None:
        print(f"   - Dim reduction: {model.feat_dim} -> 512")
    
    # Create dummy input
    n_way = 5
    k_shot = 5
    n_query = 8
    image_size = 224
    batch_size = n_way * (k_shot + n_query)
    
    print(f"\n2. Creating input data...")
    print(f"   - Batch size: {batch_size} (n_way={n_way}, k_shot={k_shot}, n_query={n_query})")
    print(f"   - Image size: {image_size}x{image_size}")
    
    x = torch.randn(n_way, k_shot + n_query, 3, image_size, image_size, device=device)
    
    if device.type == 'cuda':
        print(f"   - GPU memory before forward: {get_memory_allocated():.2f} GB")
    
    # Forward pass
    print(f"\n3. Running forward pass...")
    try:
        with torch.no_grad():
            scores = model.set_forward(x, is_feature=False)
        
        print(f"   ✓ Forward pass successful!")
        print(f"   - Output shape: {scores.shape}")
        print(f"   - Expected shape: ({n_way * n_query}, {n_way})")
        
        if device.type == 'cuda':
            print(f"   - GPU memory after forward: {get_memory_allocated():.2f} GB")
        
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"   ✗ OOM Error: {e}")
            return False
        else:
            raise e

def test_backward_pass():
    """Test backward pass (training)"""
    print("\n" + "=" * 60)
    print("Testing Backward Pass (Training)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate ResNet34 feature extractor
    class ResNet34FeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.final_feat_dim = 512 * 7 * 7
            
        def forward(self, x):
            batch_size = x.size(0)
            return torch.randn(batch_size, self.final_feat_dim, device=x.device)
    
    print("\n1. Creating model for training...")
    model = EnhancedFewShotTransformer(
        model_func=ResNet34FeatureModel,
        n_way=5,
        k_shot=5,
        n_query=8,
        variant='cosine',
        reduced_dim=512,
        use_mahalanobis=True,
        use_vic=True,
        use_checkpoint=True  # Enable gradient checkpointing for memory efficiency
    ).to(device)
    
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Create dummy input
    n_way = 5
    k_shot = 5
    n_query = 8
    x = torch.randn(n_way, k_shot + n_query, 3, 224, 224, device=device)
    
    if device.type == 'cuda':
        print(f"   - GPU memory before training step: {get_memory_allocated():.2f} GB")
    
    print(f"\n2. Running training step (forward + backward)...")
    try:
        optimizer.zero_grad()
        
        # Forward pass
        acc, loss = model.set_forward_loss(x)
        
        print(f"   ✓ Forward pass successful!")
        print(f"   - Accuracy: {acc*100:.2f}%")
        print(f"   - Loss: {loss.item():.4f}")
        
        if device.type == 'cuda':
            print(f"   - GPU memory after forward: {get_memory_allocated():.2f} GB")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"   ✓ Backward pass successful!")
        
        if device.type == 'cuda':
            print(f"   - GPU memory after backward: {get_memory_allocated():.2f} GB")
        
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"   ✗ OOM Error: {e}")
            return False
        else:
            raise e

def test_mahalanobis_memory():
    """Test Mahalanobis classifier memory usage directly"""
    print("\n" + "=" * 60)
    print("Testing Mahalanobis Classifier Memory Usage")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    classifier = MahalanobisClassifier().to(device)
    
    # Test with different dimensions
    test_cases = [
        (512, "Reduced dimension (512)"),
        (1024, "Medium dimension (1024)"),
        (2048, "Large dimension (2048)"),
    ]
    
    all_success = True
    for dim, desc in test_cases:
        print(f"\n{desc}:")
        n_way = 5
        k_shot = 5
        n_query = 40
        
        query_embeddings = torch.randn(n_query, dim, device=device)
        support_embeddings = torch.randn(n_way, k_shot, dim, device=device)
        prototypes = torch.randn(n_way, dim, device=device)
        
        try:
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                mem_before = get_memory_allocated()
            
            logits = classifier(query_embeddings, support_embeddings, prototypes)
            
            if device.type == 'cuda':
                mem_after = get_memory_allocated()
                mem_peak = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"   ✓ Success! Peak memory: {mem_peak:.3f} GB, Delta: {mem_after - mem_before:.3f} GB")
            else:
                print(f"   ✓ Success! Output shape: {logits.shape}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   ✗ OOM at dimension {dim}")
                all_success = False
            else:
                raise e
    
    return all_success

def main():
    print("\n" + "=" * 60)
    print("MEMORY OPTIMIZATION TEST SUITE")
    print("=" * 60)
    print("\nThis test verifies that the dimensionality reduction fix")
    print("prevents CUDA OOM errors when using Mahalanobis classifier")
    print("with high-dimensional features (e.g., ResNet34).\n")
    
    # Run tests
    results = []
    
    # Test 1: Forward pass
    results.append(("Forward Pass", test_forward_pass()))
    
    # Test 2: Backward pass
    results.append(("Backward Pass", test_backward_pass()))
    
    # Test 3: Mahalanobis memory
    results.append(("Mahalanobis Memory", test_mahalanobis_memory()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed! Memory optimization is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
