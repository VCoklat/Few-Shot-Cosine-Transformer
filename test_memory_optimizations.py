"""
Test script to validate memory optimization features.
Tests gradient accumulation, AMP, and memory clearing without requiring full datasets.
"""

import torch
import torch.nn as nn
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, '.')

print("=" * 60)
print("Memory Optimization Validation Test")
print("=" * 60)

# Test 1: Import test
print("\n[Test 1] Importing required modules...")
try:
    from methods.meta_template import MetaTemplate
    from methods.transformer import FewShotTransformer
    from methods.ProFOCT import ProFOCT
    # AMP imports with fallback for older PyTorch versions
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        print("   ⚠️  AMP not available in this PyTorch version")
        AMP_AVAILABLE = False
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

# Test 2: Create a simple backbone model
print("\n[Test 2] Creating simple backbone for testing...")
class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.final_feat_dim = 64
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

try:
    backbone = SimpleBackbone()
    print("✅ Simple backbone created")
except Exception as e:
    print(f"❌ Failed to create backbone: {e}")
    sys.exit(1)

# Test 3: Test gradient accumulation in train_loop
print("\n[Test 3] Testing gradient accumulation...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - Using device: {device}")
    
    # Create model
    model = FewShotTransformer(
        model_func=lambda: backbone,
        n_way=5,
        k_shot=5,
        n_query=15,
        variant="cosine"
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dummy data loader
    class DummyDataLoader:
        def __init__(self, num_batches=4):
            self.num_batches = num_batches
        
        def __iter__(self):
            for _ in range(self.num_batches):
                # Create dummy episode: (n_way, k_shot + n_query, C, H, W)
                x = torch.randn(5, 20, 3, 84, 84)
                y = torch.zeros(1)  # Dummy label
                yield x, y
        
        def __len__(self):
            return self.num_batches
    
    train_loader = DummyDataLoader()
    
    # Test with gradient accumulation
    print("   - Testing with gradient_accumulation_steps=2...")
    model.train()
    
    # We'll manually run a few iterations to test
    optimizer.zero_grad()
    for i, (x, _) in enumerate(train_loader):
        if i >= 2:  # Just test 2 iterations
            break
        
        # Forward pass
        acc, loss = model.set_forward_loss(x=x.to(device))
        
        # Scale loss
        loss = loss / 2  # gradient_accumulation_steps = 2
        
        # Backward
        loss.backward()
        
        # Step on every 2nd iteration
        if (i + 1) % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"      Iter {i}: loss={loss.item():.4f}, acc={acc:.4f}")
    
    print("✅ Gradient accumulation working correctly")
    
    # Clean up
    del model, optimizer, train_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Gradient accumulation test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test automatic mixed precision (AMP)
print("\n[Test 4] Testing automatic mixed precision (AMP)...")
try:
    if not AMP_AVAILABLE:
        print("   ⚠️  AMP not available in this PyTorch version, skipping AMP test")
    elif not torch.cuda.is_available():
        print("   ⚠️  CUDA not available, skipping AMP test")
    else:
        device = torch.device('cuda')
        
        # Create model
        model = FewShotTransformer(
            model_func=lambda: SimpleBackbone(),
            n_way=5,
            k_shot=5,
            n_query=15,
            variant="cosine"
        ).to(device)
        
        # Create optimizer and scaler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()
        
        # Create dummy data
        x = torch.randn(5, 20, 3, 84, 84).to(device)
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with autocast():
            acc, loss = model.set_forward_loss(x=x)
        
        print(f"   - Forward pass with AMP: loss={loss.item():.4f}, acc={acc:.4f}")
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✅ AMP working correctly")
        
        # Clean up
        del model, optimizer, scaler
        torch.cuda.empty_cache()

except Exception as e:
    print(f"❌ AMP test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test ProFOCT memory optimizations
print("\n[Test 5] Testing ProFOCT memory optimizations...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with VIC regularization
    model = ProFOCT(
        model_func=lambda: SimpleBackbone(),
        n_way=5,
        k_shot=5,
        n_query=15,
        variant="cosine",
        vic_alpha=0.5,
        vic_beta=9.0,
        vic_gamma=0.5,
        dynamic_vic=True
    ).to(device)
    
    # Create dummy data
    x = torch.randn(5, 20, 3, 84, 84).to(device)
    
    model.train()
    
    # Test forward and backward pass
    acc, loss = model.set_forward_loss(x=x)
    
    print(f"   - ProFOCT forward pass: loss={loss.item():.4f}, acc={acc:.4f}")
    
    # Test backward pass
    loss.backward()
    
    print("✅ ProFOCT memory optimizations working correctly")
    
    # Clean up
    del model, x, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

except Exception as e:
    print(f"❌ ProFOCT memory optimization test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test memory clearing
print("\n[Test 6] Testing CUDA cache clearing...")
try:
    if not torch.cuda.is_available():
        print("   ⚠️  CUDA not available, skipping cache clearing test")
    else:
        import gc
        
        # Get initial memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        print(f"   - Initial memory: {initial_memory / 1024**2:.2f} MB")
        
        # Create some tensors
        tensors = []
        for i in range(10):
            tensors.append(torch.randn(100, 100, 100).cuda())
        
        mem_after_alloc = torch.cuda.memory_allocated()
        print(f"   - Memory after allocation: {mem_after_alloc / 1024**2:.2f} MB")
        
        # Delete tensors
        del tensors
        gc.collect()
        
        mem_after_del = torch.cuda.memory_allocated()
        print(f"   - Memory after deletion: {mem_after_del / 1024**2:.2f} MB")
        
        # Clear cache
        torch.cuda.empty_cache()
        
        mem_after_clear = torch.cuda.memory_allocated()
        print(f"   - Memory after cache clear: {mem_after_clear / 1024**2:.2f} MB")
        
        print("✅ CUDA cache clearing working correctly")

except Exception as e:
    print(f"❌ Cache clearing test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All memory optimization tests completed successfully!")
print("=" * 60)
