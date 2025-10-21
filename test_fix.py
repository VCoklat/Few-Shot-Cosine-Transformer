"""
Test script to verify the fix for the matrix multiplication error.
This script reproduces the scenario from the problem statement.
"""
import torch
import torch.nn as nn
from methods.transformer import FewShotTransformer, Attention

def test_original_issue():
    """
    Test that reproduces the original error:
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (40x64 and 512x512)
    """
    print("Testing fix for original matrix multiplication error...")
    print("=" * 60)
    
    # Parameters from the error traceback
    n_way = 5
    k_shot = 5
    n_query = 15
    
    # Create a simple feature extractor
    def feature_model():
        class SimpleFeature(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.final_feat_dim = 512  # Common ResNet dimension
                
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                # Pad to 512 dimensions
                x = x.flatten(1)
                x = torch.cat([x, torch.zeros(x.shape[0], 512 - x.shape[1]).to(x.device)], dim=1)
                return x
        return SimpleFeature()
    
    try:
        # Create model with default parameters (heads=8, dim_head=64)
        model = FewShotTransformer(
            feature_model, 
            n_way=n_way, 
            k_shot=k_shot, 
            n_query=n_query,
            variant="cosine",
            heads=8,
            dim_head=64
        )
        print("✓ Model created successfully")
        
        # Test forward pass
        x = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
        output = model.set_forward(x, is_feature=False)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: ({n_way * n_query}, {n_way})")
        
        # Test set_forward_loss
        acc, loss = model.set_forward_loss(x)
        print(f"✓ Loss computation successful!")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Loss: {loss.item():.4f}")
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED - Original issue is fixed!")
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
    success = test_original_issue()
    exit(0 if success else 1)
