# Enhanced Few-Shot Learning with Task-Adaptive Invariance

This implementation adds invariance modules to improve few-shot learning performance, particularly in 1-shot scenarios and domain-specific tasks.

## Overview

The enhanced model integrates multiple invariance mechanisms:

1. **Task-Adaptive Invariance**: Learns task-specific invariance transformations
2. **Multi-Scale Invariance**: Processes features at multiple scales for fine-grained recognition
3. **Feature-Level Augmentation**: Augments features to improve 1-shot robustness
4. **Prototypical Refinement**: Iteratively refines prototypes using query information
5. **Medical Image Invariance**: Domain-specific invariance for medical imaging

## Architecture

```
├── modules/
│   ├── __init__.py
│   ├── task_invariance.py          # Task-adaptive and multi-scale invariance
│   ├── feature_augmentation.py     # Feature augmentation and prototype refinement
│   └── medical_invariance.py       # Medical domain-specific invariance
├── models/
│   ├── __init__.py
│   └── optimal_fewshot_enhanced.py # Enhanced model integrating all modules
└── train_enhanced.py               # Training script with invariance learning
```

## Modules

### 1. Task-Adaptive Invariance (`modules/task_invariance.py`)

#### `TaskAdaptiveInvariance`
- Learns multiple invariance transformation types
- Uses task-conditioned attention to select relevant transformations
- Applies residual connections with learnable scaling

#### `MultiScaleInvariance`
- Processes features at multiple scales
- Applies scale-specific attention
- Fuses multi-scale information adaptively

### 2. Feature-Level Augmentation (`modules/feature_augmentation.py`)

#### `FeatureLevelAugmentation`
- Learns perturbation directions in feature space
- Predicts task-adaptive magnitude
- Generates and mixes augmented features

#### `PrototypicalRefinement`
- Iteratively refines prototypes using query features
- Uses attention to select relevant query information
- Applies learnable step sizes per iteration

### 3. Medical Image Invariance (`modules/medical_invariance.py`)

#### `MedicalImageInvariance`
- **Color/Intensity pathway**: Handles lighting and staining variations
- **Texture pathway**: Preserves fine-grained texture patterns
- **Shape pathway**: Preserves structural features
- Adaptive fusion based on input

#### `ContrastiveInvarianceLoss`
- Enforces invariance through contrastive learning
- Pulls together augmented views
- Pushes apart different classes

## Dataset-Specific Configurations

The model automatically configures itself based on the dataset:

### Omniglot
```python
{
    'use_task_invariance': True,
    'use_multi_scale': False,
    'use_feature_augmentation': True,
    'use_prototype_refinement': False,
    'domain': 'general',
    'dropout': 0.05
}
```

### miniImageNet
```python
{
    'use_task_invariance': True,
    'use_multi_scale': True,
    'use_feature_augmentation': True,
    'use_prototype_refinement': True,
    'domain': 'general',
    'dropout': 0.1
}
```

### HAM10000
```python
{
    'use_task_invariance': True,
    'use_multi_scale': True,
    'use_feature_augmentation': True,
    'use_prototype_refinement': True,
    'domain': 'medical',
    'dropout': 0.2
}
```

### CUB
```python
{
    'use_task_invariance': True,
    'use_multi_scale': True,
    'use_feature_augmentation': True,
    'use_prototype_refinement': True,
    'domain': 'fine_grained',
    'dropout': 0.15
}
```

## Usage

### Training

```bash
# Train on miniImageNet 1-shot with Conv4
python train_enhanced.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 1 \
    --n_query 16 \
    --num_epoch 100 \
    --learning_rate 1e-3

# Train on HAM10000 5-shot with ResNet18
python train_enhanced.py \
    --dataset HAM10000 \
    --backbone ResNet18 \
    --n_way 7 \
    --k_shot 5 \
    --n_query 16 \
    --num_epoch 100 \
    --learning_rate 5e-4

# Train on CUB 1-shot with ResNet34
python train_enhanced.py \
    --dataset CUB \
    --backbone ResNet34 \
    --n_way 5 \
    --k_shot 1 \
    --n_query 16 \
    --num_epoch 100 \
    --learning_rate 5e-4
```

### Using in Code

```python
from models.optimal_fewshot_enhanced import get_model_for_dataset
import backbone

# Create model with automatic configuration
model = get_model_for_dataset(
    dataset='miniImagenet',
    model_func=backbone.Conv4,
    n_way=5,
    k_shot=1,
    n_query=16,
    feature_dim=64,
    n_heads=4,
    dropout=0.1
)

# Or create with custom configuration
from models.optimal_fewshot_enhanced import EnhancedOptimalFewShot

model = EnhancedOptimalFewShot(
    model_func=backbone.ResNet18,
    n_way=5,
    k_shot=1,
    n_query=16,
    feature_dim=64,
    use_task_invariance=True,
    use_multi_scale=True,
    use_feature_augmentation=True,
    use_prototype_refinement=True,
    domain='medical'
)
```

## Expected Improvements

Based on the problem statement, the following improvements are targeted:

### miniImageNet
- **1-shot ResNet34**: Target +30.21% improvement (65.15% → 95.36%)
- **1-shot Conv4**: Target +0.98% improvement

### HAM10000
- **1-shot (all backbones)**: Target +0.36-0.47% improvement
- **5-shot Conv4**: Target +2.55% improvement

### CUB
- **1-shot (all backbones)**: Target +0.56-1.10% improvement

## Key Features

1. **Gradient Clipping**: Stabilizes training with clipping value of 1.0
2. **Cosine Annealing**: Learning rate schedule for better convergence
3. **Differential Learning Rates**: Lower LR for backbone, higher for invariance modules
4. **Gradient Checkpointing**: Memory-efficient training
5. **Automatic Dataset Configuration**: Optimal settings per dataset

## Training Details

- **Optimizer**: AdamW (default), also supports Adam and SGD
- **Learning Rate**: 1e-3 (default), with 0.1x for backbone
- **Weight Decay**: 1e-5
- **Gradient Clipping**: 1.0
- **Scheduler**: CosineAnnealingLR with eta_min=1e-6

## Testing

After training, test the model using the standard test script:

```bash
python test.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --method EnhancedOptimalFewShot \
    --n_way 5 \
    --k_shot 1 \
    --test_iter 600
```

## Implementation Notes

1. **Memory Efficiency**: Uses gradient checkpointing to reduce memory usage
2. **Stability**: Residual connections with small initial gamma (0.1)
3. **Flexibility**: Modular design allows easy addition/removal of components
4. **Compatibility**: Inherits from base `OptimalFewShotModel` for compatibility

## Citation

If you use this code, please cite the original Few-Shot-Cosine-Transformer repository.

## License

This implementation follows the license of the parent repository.
