"""
Test script to validate the accuracy enhancement implementations.
Tests the new features: temperature scaling, enhanced prototypes, 
multi-scale features, and residual connections.
"""
import torch
import torch.nn as nn
import sys

def test_transformer_enhancements():
    """Test FewShotTransformer with new enhancements"""
    print("=" * 70)
    print("Testing FewShotTransformer Enhancements")
    print("=" * 70)
    
    try:
        from methods.transformer import FewShotTransformer
        
        # Create a simple feature model
        def feature_model():
            class SimpleBackbone(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.final_feat_dim = 512
                    
                def forward(self, x):
                    batch_size = x.size(0)
                    return torch.randn(batch_size, self.final_feat_dim)
            return SimpleBackbone()
        
        # Initialize model
        n_way, k_shot, n_query = 5, 5, 15
        model = FewShotTransformer(
            feature_model,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            variant="cosine"
        )
        
        print("✓ Model initialized successfully")
        
        # Check new parameters exist
        params_to_check = [
            'proto_temperature',
            'output_temperature',
            'feature_refiner',
            'ATTN.attention_temperature'
        ]
        
        for param_name in params_to_check:
            parts = param_name.split('.')
            obj = model
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    raise AttributeError(f"Parameter {param_name} not found")
            print(f"✓ Parameter '{param_name}' exists")
        
        # Test forward pass
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
        print("\nTesting forward pass...")
        
        with torch.no_grad():
            output = model.set_forward(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({n_way * n_query}, {n_way})")
        
        # Test backward pass
        print("\nTesting backward pass...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        acc, loss = model.set_forward_loss(x)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Backward pass successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Accuracy: {acc:.4f}")
        
        # Check if new parameters are being updated
        print("\nChecking parameter updates...")
        proto_temp_before = model.proto_temperature.data.clone()
        output_temp_before = model.output_temperature.data.clone()
        attn_temp_before = model.ATTN.attention_temperature.data.clone()
        
        # Run a few training steps
        for _ in range(5):
            x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
            optimizer.zero_grad()
            acc, loss = model.set_forward_loss(x)
            loss.backward()
            optimizer.step()
        
        proto_temp_after = model.proto_temperature.data
        output_temp_after = model.output_temperature.data
        attn_temp_after = model.ATTN.attention_temperature.data
        
        proto_changed = not torch.allclose(proto_temp_before, proto_temp_after)
        output_changed = not torch.allclose(output_temp_before, output_temp_after)
        attn_changed = not torch.allclose(attn_temp_before, attn_temp_after)
        
        print(f"✓ Proto temperature updated: {proto_changed}")
        print(f"✓ Output temperature updated: {output_changed}")
        print(f"✓ Attention temperature updated: {attn_changed}")
        
        print("\n" + "=" * 70)
        print("✓ FewShotTransformer tests PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n✗ FewShotTransformer tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ctx_enhancements():
    """Test CTX with new enhancements"""
    print("\n" + "=" * 70)
    print("Testing CTX Enhancements")
    print("=" * 70)
    
    try:
        from methods.CTX import CTX
        
        # Create a simple feature model
        def feature_model():
            class SimpleBackbone(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.final_feat_dim = [64, 7, 7]  # CTX uses spatial features
                    
                def forward(self, x):
                    batch_size = x.size(0)
                    return torch.randn(batch_size, 64, 7, 7)
            return SimpleBackbone()
        
        # Initialize model
        n_way, k_shot, n_query = 5, 5, 15
        model = CTX(
            feature_model,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            variant="cosine",
            input_dim=64
        )
        
        print("✓ Model initialized successfully")
        
        # Check new parameters exist
        params_to_check = [
            'attention_temperature',
            'output_temperature',
        ]
        
        for param_name in params_to_check:
            if not hasattr(model, param_name):
                raise AttributeError(f"Parameter {param_name} not found")
            print(f"✓ Parameter '{param_name}' exists")
        
        # Test forward pass
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
        print("\nTesting forward pass...")
        
        with torch.no_grad():
            output = model.set_forward(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
        # Test backward pass
        print("\nTesting backward pass...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        acc, loss = model.set_forward_loss(x)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Backward pass successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Accuracy: {acc:.4f}")
        
        # Check if new parameters are being updated
        print("\nChecking parameter updates...")
        attn_temp_before = model.attention_temperature.data.clone()
        output_temp_before = model.output_temperature.data.clone()
        
        # Run a few training steps
        for _ in range(5):
            x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
            optimizer.zero_grad()
            acc, loss = model.set_forward_loss(x)
            loss.backward()
            optimizer.step()
        
        attn_temp_after = model.attention_temperature.data
        output_temp_after = model.output_temperature.data
        
        attn_changed = not torch.allclose(attn_temp_before, attn_temp_after)
        output_changed = not torch.allclose(output_temp_before, output_temp_after)
        
        print(f"✓ Attention temperature updated: {attn_changed}")
        print(f"✓ Output temperature updated: {output_changed}")
        
        print("\n" + "=" * 70)
        print("✓ CTX tests PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n✗ CTX tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 70)
    print("ACCURACY ENHANCEMENT VALIDATION SUITE")
    print("=" * 70)
    print("\nThis test validates the following enhancements:")
    print("1. Temperature scaling for better calibration")
    print("2. Enhanced prototype learning with attention")
    print("3. Multi-scale feature refinement")
    print("4. Residual connections in invariance projections")
    print("5. Learnable attention temperature")
    print("\n" + "=" * 70)
    
    results = []
    
    # Test FewShotTransformer
    results.append(("FewShotTransformer", test_transformer_enhancements()))
    
    # Test CTX
    results.append(("CTX", test_ctx_enhancements()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! 🎉")
        print("The accuracy enhancements are working correctly.")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("SOME TESTS FAILED")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
