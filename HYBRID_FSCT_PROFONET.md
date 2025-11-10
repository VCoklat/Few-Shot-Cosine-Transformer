# Hybrid FS-CT + ProFONet Implementation

This document describes the implementation of the hybrid Few-Shot Cosine Transformer (FS-CT) + ProFONet algorithm with VIC regularization and memory optimization features.

## Overview

The hybrid algorithm integrates three key innovations:

1. **Learnable Prototypical Embedding** (from FS-CT) with **VIC Regularization** (from ProFONet)
2. **Cosine Attention Transformer** (from FS-CT)
3. **Dynamic Weight VIC** for memory-efficient training

## New Features

### 1. VIC Regularization

VIC (Variance-Invariance-Covariance) regularization helps prevent representation collapse and improves feature quality:

- **Variance Regularization**: Prevents norm collapse by penalizing features with low variance
- **Invariance Regularization**: Maintains classification accuracy (cross-entropy loss)
- **Covariance Regularization**: Prevents representation collapse by penalizing correlated features

#### Dynamic Weight Adjustment

The weights for VIC components adjust automatically during training:

```python
# At epoch 0
λ_V = 0.5   # Variance weight
λ_I = 9.0   # Invariance weight (constant)
λ_C = 0.5   # Covariance weight

# At epoch 50
λ_V = 0.65  # Increases during training
λ_I = 9.0   # Remains constant
λ_C = 0.40  # Decreases during training
```

### 2. Memory Optimization Features

#### Gradient Checkpointing

Reduces memory usage by recomputing intermediate activations during backward pass:

```bash
python train.py --gradient_checkpointing 1 --backbone ResNet18
```

Memory savings: ~30-40% for ResNet models

#### Mixed Precision Training

Uses FP16 for forward/backward passes while maintaining FP32 for critical operations:

```bash
python train.py --mixed_precision 1
```

Memory savings: ~40-50%, training speedup: ~2-3x

#### Gradient Clipping

Improves training stability by limiting gradient norm:

```bash
python train.py --gradient_clip 1.0
```

## Usage

### Basic Usage with VIC Regularization

```bash
python train.py \
  --method FSCT_cosine \
  --dataset miniImagenet \
  --backbone ResNet18 \
  --n_way 5 \
  --k_shot 5 \
  --use_vic 1
```

### Full Memory-Optimized Configuration

```bash
python train.py \
  --method FSCT_cosine \
  --dataset miniImagenet \
  --backbone ResNet18 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 10 \
  --use_vic 1 \
  --mixed_precision 1 \
  --gradient_checkpointing 1 \
  --gradient_clip 1.0
```

## Command Line Arguments

### VIC Regularization Parameters

- `--use_vic`: Enable VIC regularization (0 or 1, default: 0)
- `--lambda_V_base`: Base weight for variance regularization (default: 0.5)
- `--lambda_I`: Weight for invariance loss (default: 9.0)
- `--lambda_C_base`: Base weight for covariance regularization (default: 0.5)
- `--vic_gamma`: Gamma parameter for variance regularization (default: 1.0)
- `--vic_epsilon`: Epsilon parameter for variance regularization (default: 1e-6)

### Memory Optimization Parameters

- `--mixed_precision`: Use mixed precision training (0 or 1, default: 0)
- `--gradient_checkpointing`: Use gradient checkpointing (0 or 1, default: 0)
- `--gradient_clip`: Gradient clipping max norm (default: 1.0, set to 0 to disable)

### Reduced Resource Configuration

For 16GB VRAM constraint:

```bash
python train.py \
  --method FSCT_cosine \
  --backbone Conv6 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 10 \
  --use_vic 1 \
  --mixed_precision 1 \
  --gradient_checkpointing 1
```

## Architecture Details

### VIC Regularization Module

Located in `methods/vic_regularization.py`:

```python
from methods.vic_regularization import VICRegularization, DynamicVICWeights

# Initialize VIC regularization
vic_reg = VICRegularization(gamma=1.0, epsilon=1e-6)

# Compute losses
vic_losses = vic_reg(embeddings)
# Returns: {'variance_loss': tensor, 'covariance_loss': tensor}

# Dynamic weights
vic_weights = DynamicVICWeights(
    lambda_V_base=0.5,
    lambda_I=9.0,
    lambda_C_base=0.5
)
weights = vic_weights.get_weights(current_epoch, total_epochs)
```

### Integration with FewShotTransformer

The VIC regularization is integrated into `FewShotTransformer` class:

```python
model = FewShotTransformer(
    feature_model,
    n_way=5,
    k_shot=5,
    n_query=16,
    variant='cosine',
    use_vic=True,
    lambda_V_base=0.5,
    lambda_I=9.0,
    lambda_C_base=0.5
)
```

## Performance Expectations

With VIC regularization and memory optimizations:

- **Accuracy Improvement**: Expected >5% improvement over baseline FS-CT
- **Memory Usage**: ~40-50% reduction with all optimizations enabled
- **Training Speed**: ~10-20% slower with gradient checkpointing, ~2x faster with mixed precision

## Testing

Run the VIC regularization tests:

```bash
python test_vic_regularization.py
```

Expected output:
```
============================================================
Running VIC Regularization Tests
============================================================
Testing VIC Regularization basic functionality...
  ✓ VIC Regularization basic functionality test passed
...
============================================================
✓ All VIC Regularization tests passed!
============================================================
```

## Implementation Files

- `methods/vic_regularization.py`: VIC regularization module
- `methods/transformer.py`: Updated FewShotTransformer with VIC integration
- `methods/meta_template.py`: Updated training loop for dynamic weights
- `backbone.py`: Added gradient checkpointing support
- `train.py`: Updated training script with memory optimizations
- `io_utils.py`: Added new command line arguments
- `test_vic_regularization.py`: Unit tests for VIC module

## References

- **FS-CT Paper**: "Enhancing Few-shot Image Classification with Cosine Transformer" (IEEE Access 2023)
- **ProFONet**: VIC regularization for few-shot learning
- **Memory Optimization**: PyTorch gradient checkpointing and mixed precision training

## Notes

- VIC regularization is only applied during training, not during validation/testing
- Gradient checkpointing only works with ResNet backbones in FETI mode
- Mixed precision training requires CUDA-capable GPU
- The implementation maintains backward compatibility with existing FS-CT methods
