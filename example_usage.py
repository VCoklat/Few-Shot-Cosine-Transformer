#!/usr/bin/env python3
"""
Example usage of Enhanced Few-Shot Learning Models

This script demonstrates how to use the enhanced models with different
configurations and datasets.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def example_basic_usage():
    """Example: Basic usage with default configuration"""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    try:
        import torch
        from models.optimal_fewshot_enhanced import get_model_for_dataset
        
        # Create model for miniImageNet with automatic configuration
        model = get_model_for_dataset(
            dataset='miniimagenet',
            model_func=lambda: None,  # Will use default Conv4
            n_way=5,
            k_shot=1,
            n_query=16
        )
        
        print(f"✓ Created model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
        print(f"  - Task-adaptive invariance: {model.use_task_invariance}")
        print(f"  - Multi-scale invariance: {model.use_multi_scale}")
        print(f"  - Feature augmentation: {model.use_feature_augmentation}")
        print(f"  - Prototype refinement: {model.use_prototype_refinement}")
        print(f"  - Domain: {model.domain}")
        
    except ImportError as e:
        print(f"⚠ Skipping example (missing dependencies): {e}")
    except Exception as e:
        print(f"✗ Example failed: {e}")
        import traceback
        traceback.print_exc()


def example_custom_configuration():
    """Example: Custom configuration"""
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)
    
    try:
        import torch
        from models.optimal_fewshot_enhanced import EnhancedOptimalFewShot
        
        # Create model with custom configuration
        model = EnhancedOptimalFewShot(
            model_func=lambda: None,
            n_way=5,
            k_shot=1,
            n_query=16,
            feature_dim=128,  # Larger feature dimension
            n_heads=8,  # More attention heads
            dropout=0.15,
            use_task_invariance=True,
            use_multi_scale=True,
            use_feature_augmentation=True,
            use_prototype_refinement=False,  # Disable for faster training
            domain='general'
        )
        
        print("✓ Created custom model")
        print(f"  - Feature dim: {model.transformer_dim}")
        print(f"  - Dropout: 0.15")
        print(f"  - Prototype refinement: disabled")
        
    except ImportError as e:
        print(f"⚠ Skipping example (missing dependencies): {e}")
    except Exception as e:
        print(f"✗ Example failed: {e}")


def example_medical_imaging():
    """Example: Medical imaging configuration"""
    print("\n" + "=" * 80)
    print("Example 3: Medical Imaging (HAM10000)")
    print("=" * 80)
    
    try:
        import torch
        from models.optimal_fewshot_enhanced import get_model_for_dataset
        
        # Create model for HAM10000 with medical-specific invariance
        model = get_model_for_dataset(
            dataset='ham10000',
            model_func=lambda: None,
            n_way=7,  # HAM10000 has 7 classes
            k_shot=5,
            n_query=16
        )
        
        print("✓ Created medical imaging model")
        print(f"  - Domain: {model.domain}")
        print(f"  - Has medical invariance: {hasattr(model, 'medical_invariance')}")
        print(f"  - Has contrastive loss: {hasattr(model, 'contrastive_loss')}")
        print(f"  - Multi-scale: {model.use_multi_scale}")
        
    except ImportError as e:
        print(f"⚠ Skipping example (missing dependencies): {e}")
    except Exception as e:
        print(f"✗ Example failed: {e}")


def example_fine_grained_recognition():
    """Example: Fine-grained recognition configuration"""
    print("\n" + "=" * 80)
    print("Example 4: Fine-Grained Recognition (CUB)")
    print("=" * 80)
    
    try:
        import torch
        from models.optimal_fewshot_enhanced import get_model_for_dataset
        
        # Create model for CUB with fine-grained specific settings
        model = get_model_for_dataset(
            dataset='cub',
            model_func=lambda: None,
            n_way=5,
            k_shot=1,
            n_query=16
        )
        
        print("✓ Created fine-grained recognition model")
        print(f"  - Domain: {model.domain}")
        print(f"  - Multi-scale: {model.use_multi_scale} (important for fine details)")
        print(f"  - Feature augmentation: {model.use_feature_augmentation}")
        
    except ImportError as e:
        print(f"⚠ Skipping example (missing dependencies): {e}")
    except Exception as e:
        print(f"✗ Example failed: {e}")


def example_forward_pass():
    """Example: Forward pass with dummy data"""
    print("\n" + "=" * 80)
    print("Example 5: Forward Pass")
    print("=" * 80)
    
    try:
        import torch
        from models.optimal_fewshot_enhanced import get_model_for_dataset
        
        # Create model
        model = get_model_for_dataset(
            dataset='miniimagenet',
            model_func=lambda: None,
            n_way=5,
            k_shot=1,
            n_query=16
        )
        
        model.eval()
        
        # Create dummy episode data
        # Shape: [n_way, k_shot + n_query, channels, height, width]
        n_way = 5
        k_shot = 1
        n_query = 16
        dummy_episode = torch.randn(n_way, k_shot + n_query, 3, 84, 84)
        
        print("✓ Created dummy episode data")
        print(f"  - Shape: {dummy_episode.shape}")
        
        # Forward pass
        with torch.no_grad():
            logits = model.set_forward(dummy_episode)
        
        print("✓ Forward pass successful")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Expected: [{n_way * n_query}, {n_way}]")
        
        # Compute predictions
        predictions = torch.argmax(logits, dim=1)
        print(f"  - Predictions shape: {predictions.shape}")
        
    except ImportError as e:
        print(f"⚠ Skipping example (missing dependencies): {e}")
    except Exception as e:
        print(f"✗ Example failed: {e}")
        import traceback
        traceback.print_exc()


def example_training_setup():
    """Example: Training setup"""
    print("\n" + "=" * 80)
    print("Example 6: Training Setup")
    print("=" * 80)
    
    try:
        import torch
        from models.optimal_fewshot_enhanced import get_model_for_dataset
        
        # Create model
        model = get_model_for_dataset(
            dataset='miniimagenet',
            model_func=lambda: None,
            n_way=5,
            k_shot=1,
            n_query=16
        )
        
        # Setup optimizer with differential learning rates
        backbone_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'feature' in name:  # Backbone
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-4},  # Lower LR for backbone
            {'params': other_params, 'lr': 1e-3}      # Higher LR for invariance modules
        ], weight_decay=1e-5)
        
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        
        print("✓ Training setup complete")
        print(f"  - Backbone params: {len(backbone_params)}")
        print(f"  - Other params: {len(other_params)}")
        print(f"  - Optimizer: AdamW with differential LR")
        print(f"  - Scheduler: CosineAnnealingLR")
        
    except ImportError as e:
        print(f"⚠ Skipping example (missing dependencies): {e}")
    except Exception as e:
        print(f"✗ Example failed: {e}")


def main():
    """Run all examples"""
    print("=" * 80)
    print("Enhanced Few-Shot Learning - Usage Examples")
    print("=" * 80)
    print()
    
    examples = [
        example_basic_usage,
        example_custom_configuration,
        example_medical_imaging,
        example_fine_grained_recognition,
        example_forward_pass,
        example_training_setup
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Example failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - ENHANCED_MODEL_README.md")
    print("  - train_enhanced.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
