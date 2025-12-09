"""
Test to simulate the exact failing scenario from optimal_few_shot.py line 313.
This test creates an OptimalFewShot model with ResNet34 backbone using HAM10000 dataset,
which is the exact configuration that was failing.
"""

import sys
import os
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbone import ResNet34


def test_optimal_few_shot_scenario():
    """
    Test the exact scenario from the problem statement:
    Creating a model with ResNet34 backbone and HAM10000 dataset,
    then accessing the training attribute as done in optimal_few_shot.py line 313.
    """
    print("=" * 80)
    print("Testing Exact Scenario from Problem Statement")
    print("=" * 80)
    print()
    
    print("Simulating the exact code path that was failing:")
    print("  File: methods/optimal_few_shot.py, line 313")
    print("  Code: was_training = self.feature.training")
    print()
    
    # Simulate the exact model creation from run_experiments.py
    # which calls create_model('proposed', config) which creates OptimalFewShot
    # with ResNet34 as the backbone
    
    print("Creating ResNet34 model with FETI=0 (uses ResNetModel)...")
    print("  Dataset: HAM10000")
    print("  Backbone: ResNet34")
    print()
    
    # This is what happens in call_model_func when model_func is ResNet34
    model_func = ResNet34
    dataset = 'HAM10000'
    feti = 0  # FETI=0 means it uses ResNetModel instead of ResNet
    flatten = True
    
    # Call the model function (simulating call_model_func)
    feature = model_func(feti, dataset, flatten)
    print(f"✓ Created feature extractor: {type(feature).__name__}")
    print()
    
    # This is the exact line that was failing (line 313 in optimal_few_shot.py)
    print("Accessing training attribute (line 313 of optimal_few_shot.py)...")
    try:
        was_training = feature.training  # This line was causing AttributeError
        print(f"✓ Successfully accessed training attribute: {was_training}")
        print()
        
        # Test the subsequent operations (lines 314-329)
        print("Testing subsequent operations from optimal_few_shot.py...")
        
        # Line 314: Set to eval mode
        feature.eval()
        print(f"✓ Set to eval mode: training={feature.training}")
        
        # Restore training state (as would happen later in the code)
        if was_training:
            feature.train()
        print(f"✓ Restored training state: training={feature.training}")
        print()
        
        print("=" * 80)
        print("SUCCESS: The exact failing scenario from the problem statement is now fixed!")
        print("=" * 80)
        print()
        print("The error was:")
        print("  AttributeError: 'ResNetModel' object has no attribute 'training'")
        print()
        print("The fix:")
        print("  Changed 'class ResNetModel():' to 'class ResNetModel(nn.Module):'")
        print("  in backbone.py line 237")
        print("=" * 80)
        
        return True
        
    except AttributeError as e:
        print(f"✗ FAILED with AttributeError: {e}")
        print()
        print("=" * 80)
        print("FAILED: The error from the problem statement still exists!")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        success = test_optimal_few_shot_scenario()
        sys.exit(0 if success else 1)
            
    except Exception as e:
        print()
        print("=" * 80)
        print(f"TEST FAILED with unexpected error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
