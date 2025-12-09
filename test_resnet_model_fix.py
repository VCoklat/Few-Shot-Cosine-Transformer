"""
Test to verify ResNetModel properly inherits from nn.Module and has the 'training' attribute.
This test validates the fix for the AttributeError reported in the issue.
"""

import sys
import os
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbone import ResNetModel


def test_resnet_model_has_training_attribute():
    """
    Test that ResNetModel has the 'training' attribute from nn.Module.
    This was failing before the fix with:
    AttributeError: 'ResNetModel' object has no attribute 'training'
    """
    print("=" * 80)
    print("Testing ResNetModel has 'training' attribute")
    print("=" * 80)
    print()
    
    # Create a ResNetModel instance (using HAM10000 dataset as in the error)
    print("Creating ResNetModel with HAM10000 dataset...")
    model = ResNetModel(dataset='HAM10000', variant=34, flatten=True)
    print("✓ ResNetModel created successfully")
    print()
    
    # Test that it's an instance of nn.Module
    print("Checking if ResNetModel is a nn.Module...")
    assert isinstance(model, torch.nn.Module), "ResNetModel should inherit from nn.Module"
    print("✓ ResNetModel is a nn.Module")
    print()
    
    # Test that it has the training attribute
    print("Checking if ResNetModel has 'training' attribute...")
    assert hasattr(model, 'training'), "ResNetModel should have 'training' attribute"
    print(f"✓ ResNetModel has 'training' attribute: {model.training}")
    print()
    
    # Test that we can access the training attribute
    print("Testing access to 'training' attribute...")
    was_training = model.training  # This line was failing before the fix
    print(f"✓ Successfully accessed training attribute: {was_training}")
    print()
    
    # Test that we can toggle training mode
    print("Testing training mode toggle...")
    model.eval()
    assert model.training == False, "Model should be in eval mode"
    print(f"✓ Model in eval mode: training={model.training}")
    
    model.train()
    assert model.training == True, "Model should be in training mode"
    print(f"✓ Model in training mode: training={model.training}")
    print()
    
    # Test that the model has forward method
    print("Testing forward method exists...")
    assert hasattr(model, 'forward'), "ResNetModel should have 'forward' method"
    print("✓ ResNetModel has 'forward' method")
    print()
    
    print("=" * 80)
    print("SUCCESS: All tests passed! ResNetModel is properly fixed.")
    print("=" * 80)
    
    return True


def test_resnet34_function_creates_correct_model():
    """
    Test that the ResNet34 function creates a model with the training attribute
    when FETI=False (which uses ResNetModel).
    """
    print()
    print("=" * 80)
    print("Testing ResNet34 function with FETI=False")
    print("=" * 80)
    print()
    
    from backbone import ResNet34
    
    # Create model using ResNet34 function with FETI=False (uses ResNetModel)
    print("Creating model with ResNet34(FETI=False, dataset='HAM10000', flatten=True)...")
    model = ResNet34(FETI=False, dataset='HAM10000', flatten=True)
    print("✓ Model created successfully")
    print()
    
    # Test that it has the training attribute
    print("Checking if model has 'training' attribute...")
    assert hasattr(model, 'training'), "Model should have 'training' attribute"
    print(f"✓ Model has 'training' attribute: {model.training}")
    print()
    
    print("=" * 80)
    print("SUCCESS: ResNet34 function test passed!")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    try:
        success1 = test_resnet_model_has_training_attribute()
        success2 = test_resnet34_function_creates_correct_model()
        
        if success1 and success2:
            print()
            print("=" * 80)
            print("ALL TESTS PASSED!")
            print("=" * 80)
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print()
        print("=" * 80)
        print(f"TEST FAILED with error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
