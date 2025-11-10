# VIC-Enhanced Few-Shot Cosine Transformer

This document describes the VIC (Variance-Invariance-Covariance) loss enhancement for Few-Shot Cosine Transformer (FS-CT).

## Overview

The VIC loss extends the standard training process by incorporating three loss components inspired by ProFONet:

1. **Invariance Loss (L_I)**: Standard Categorical Cross-Entropy (CCE) loss for classification
2. **Variance Loss (L_V)**: Hinge loss on standard deviation to encourage compact class representations
3. **Covariance Loss (L_C)**: Covariance regularization to decorrelate feature dimensions and prevent informational collapse

## Algorithm

### Training Loop with VIC Loss

For each episode:
1. Sample a task (episode) T = (S, Q) from the training dataset
2. Extract features Z_S and Z_Q from support and query sets
3. Calculate learnable prototypes Z_P from Z_S
4. Compute predictions ŷ using Cosine Transformer
5. Calculate combined loss:
   - L_total = (λ_I × L_I) + (λ_V × L_V) + (λ_C × L_C)
6. Update parameters via gradient descent

## Usage

### Command Line Arguments

Enable VIC loss by adding these arguments to your training command:

```bash
python train.py \
  --method FSCT_cosine \
  --dataset miniImagenet \
  --backbone ResNet18 \
  --use_vic_loss 1 \
  --lambda_v 1.0 \
  --lambda_i 1.0 \
  --lambda_c 0.04
```

### Parameters

- `--use_vic_loss`: Enable VIC loss (1 = enabled, 0 = disabled, default: 0)
- `--lambda_v`: Weight for variance loss (default: 1.0)
- `--lambda_i`: Weight for invariance (CE) loss (default: 1.0)
- `--lambda_c`: Weight for covariance loss (default: 0.04)

### Example Training Commands

#### Standard 5-way 5-shot training with VIC loss:
```bash
python train.py --method FSCT_cosine --dataset miniImagenet \
  --backbone ResNet18 --n_way 5 --k_shot 5 \
  --use_vic_loss 1 --lambda_v 1.0 --lambda_i 1.0 --lambda_c 0.04 \
  --num_epoch 50 --learning_rate 1e-3
```

#### Training with different loss weights:
```bash
python train.py --method FSCT_cosine --dataset CUB \
  --backbone ResNet34 --n_way 5 --k_shot 1 \
  --use_vic_loss 1 --lambda_v 2.0 --lambda_i 1.0 --lambda_c 0.02 \
  --num_epoch 50
```

#### Standard training (without VIC loss):
```bash
python train.py --method FSCT_cosine --dataset miniImagenet \
  --backbone ResNet18 --n_way 5 --k_shot 5 \
  --use_vic_loss 0 --num_epoch 50
```

## Memory Efficiency (8GB GPU)

The implementation is optimized for training on GPUs with 8GB VRAM:

1. **Efficient Loss Computation**: All loss components use in-place operations where possible
2. **No Redundant Storage**: Support embeddings are reused across loss components
3. **Gradient Checkpointing Compatible**: Works with PyTorch's gradient checkpointing if needed
4. **Tested Configuration**: Successfully tested with:
   - Backbone: ResNet18/ResNet34/Conv4/Conv6
   - 5-way 5-shot: ~3.5M parameters
   - Batch size: n_episode=200 episodes per epoch

## Implementation Details

### VIC Loss Module (`methods/vic_loss.py`)

The `VICLoss` class implements:

- **Invariance Loss**: Standard cross-entropy for classification task
- **Variance Loss**: Encourages compact clusters using hinge loss on standard deviation
  ```python
  hinge = max(0, threshold - std)
  ```
- **Covariance Loss**: Minimizes off-diagonal elements of covariance matrix
  ```python
  loss = sum(off_diagonal_elements^2) / (d × (d-1))
  ```

### FewShotTransformer Integration (`methods/transformer.py`)

Modified to support VIC loss:

- Added `use_vic_loss` parameter to constructor
- Modified `set_forward()` to optionally return support embeddings
- Updated `set_forward_loss()` to compute VIC loss when enabled

## Loss Weight Tuning Guidelines

Based on the ProFONet paper and our implementation:

- **λ_i** (Invariance): Start with 1.0, this is the primary classification loss
- **λ_v** (Variance): Range 0.5-2.0, higher values encourage tighter clusters
- **λ_c** (Covariance): Range 0.01-0.1, smaller values as regularization term

Recommended starting points:
- **Balanced**: λ_i=1.0, λ_v=1.0, λ_c=0.04
- **Compact clusters**: λ_i=1.0, λ_v=2.0, λ_c=0.04
- **Decorrelation focus**: λ_i=1.0, λ_v=0.5, λ_c=0.1

## Compatibility

- Works with all FS-CT methods: `FSCT_softmax`, `FSCT_cosine`
- Compatible with all backbones: `Conv4`, `Conv6`, `ResNet18`, `ResNet34`
- Supports all datasets: miniImagenet, CUB, CIFAR-FS, Omniglot, Yoga
- Can be combined with other training features (augmentation, FETI, etc.)

## References

- ProFONet: Prototypical Few-Shot Networks
- Few-Shot Cosine Transformer (FS-CT) paper
- VIC Regularization for Self-Supervised Learning
