"""
Simple test to verify validate_model function works correctly.
"""
import torch
import numpy as np
from train_test import validate_model

# Mock model class with required methods
class MockModel(torch.nn.Module):
    def __init__(self, n_way=5):
        super(MockModel, self).__init__()
        self.n_way = n_way
        self.change_way = True
        
    def correct(self, x):
        """Mock correct method that returns random accuracy"""
        query_size = 15  # typical n_query
        correct = np.random.randint(0, self.n_way * query_size)
        total = self.n_way * query_size
        return correct, total

# Mock data loader
class MockDataLoader:
    def __init__(self, num_batches=5, n_way=5):
        self.num_batches = num_batches
        self.n_way = n_way
        
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for i in range(self.num_batches):
            # Generate mock data (batch_size, channels, height, width)
            x = torch.randn(self.n_way, 3, 84, 84)
            y = torch.arange(self.n_way)
            yield x, y

def test_validate_model():
    """Test that validate_model function executes without errors"""
    print("Testing validate_model function...")
    
    # Create mock model and data loader
    model = MockModel(n_way=5)
    val_loader = MockDataLoader(num_batches=3, n_way=5)
    
    # Test the validate_model function
    try:
        val_acc = validate_model(val_loader, model)
        print(f"✓ validate_model executed successfully!")
        print(f"  Returned validation accuracy: {val_acc:.2f}%")
        
        # Check that accuracy is in valid range
        assert 0 <= val_acc <= 100, f"Invalid accuracy: {val_acc}"
        print(f"✓ Validation accuracy is in valid range [0, 100]")
        
        # Check that it returns a float
        assert isinstance(val_acc, (float, np.floating)), f"Invalid return type: {type(val_acc)}"
        print(f"✓ Returns correct type (float)")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validate_model()
    exit(0 if success else 1)
