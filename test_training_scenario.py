"""
Test that simulates the actual training scenario from the error traceback.
This ensures the fix works in the context where the error originally occurred.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from methods.transformer import FewShotTransformer

def test_training_scenario():
    """
    Simulate the training loop scenario from the error:
    train_test.py:226 -> train() -> model.train_loop() -> set_forward_loss()
    """
    print("Simulating training scenario from error traceback...")
    print("=" * 60)
    
    # Setup similar to the training script
    n_way = 5
    k_shot = 5
    n_query = 15
    
    # Create a simple feature model (like ResNet would produce)
    def feature_model():
        class SimpleBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.final_feat_dim = 512
                
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.flatten(1)
                # Simulate ResNet output dimension
                x = torch.cat([x, torch.zeros(x.shape[0], 512 - x.shape[1]).to(x.device)], dim=1)
                return x
        return SimpleBackbone()
    
    try:
        # Create model
        model = FewShotTransformer(
            feature_model,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            variant="cosine",
            depth=1,
            heads=8,
            dim_head=64,
            mlp_dim=512
        )
        print("✓ Model initialized")
        
        # Setup optimizer (like in train())
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Simulate a training batch
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
        
        # This is where the error occurred: set_forward_loss
        print("  Running forward pass through set_forward_loss...")
        optimizer.zero_grad()
        acc, loss = model.set_forward_loss(x)
        
        print(f"✓ Forward pass successful!")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Test backward pass (complete training step)
        print("  Running backward pass...")
        loss.backward()
        optimizer.step()
        print("✓ Backward pass successful!")
        
        # Test multiple iterations
        print("  Testing multiple training iterations...")
        for i in range(3):
            x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
            optimizer.zero_grad()
            acc, loss = model.set_forward_loss(x)
            loss.backward()
            optimizer.step()
        print(f"✓ Completed 3 training iterations without errors")
        
        print("=" * 60)
        print("✓ TRAINING SCENARIO TEST PASSED!")
        print("  The model can now train without dimension mismatches")
        return True
        
    except RuntimeError as e:
        if "mat1 and mat2 shapes cannot be multiplied" in str(e):
            print("✗ FAILED - Original error still present:")
            print(f"  {e}")
            return False
        else:
            print(f"✗ FAILED with different error: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"✗ FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_scenario()
    exit(0 if success else 1)
