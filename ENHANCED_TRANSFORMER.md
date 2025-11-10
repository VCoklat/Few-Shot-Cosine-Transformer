# Enhanced Few-Shot Cosine Transformer

This document describes the enhanced implementation of the Few-Shot Cosine Transformer with VIC regularization and Mahalanobis distance classifier.

## Architecture Overview

The enhanced transformer implements the following components as specified in the problem statement:

### 1. Learnable Weighted Prototypes

For each class `c` with `k` shots:
- Compute weights `w_ci` via a per-class learnable vector and softmax along shots
- Prototype: `z̄_c = Σ_i w_ci * z_ci`
- Initialized to uniform mean (zeros → softmax → uniform)

### 2. Cosine Cross-Attention

Multi-head cosine attention between prototypes (ZP) and queries (ZQ):
- **Heads (H)**: 4
- **Dimension per head (dh)**: 64
- **Encoder blocks**: 2
- **No softmax** on attention weights
- **No positional encodings** (not needed for few-shot)
- Pre-norm architecture with GELU FFN

### 3. Mahalanobis Distance Classifier

Replaces cosine linear head with prototype classifier using class-wise Mahalanobis distance:
- Distance: `D_k(x) = (x - P_k)ᵀ Σ_k⁻¹ (x - P_k)`
- Shrinkage covariance: `Σ_k = (1-α)S_k + αI`
- Adaptive shrinkage: `α ≈ d/(N_k + d)`
- Logits: `-D_k(x)` (negative distance)

### 4. VIC Regularization

Three regularization terms:

**Invariance (I)**: Classification loss
- Cross-entropy over Mahalanobis distances
- `p(y=k|x) ∝ exp(-D_k(x))`

**Variance (V)**: Per-dimension std regularization
- Hinge loss toward target `σ=1`
- `V = (1/d) Σ_j max(0, 1 - σ_j - ε)`
- Computed on concatenated `[ES, P]`

**Covariance (C)**: Decorrelation loss
- Off-diagonal squared Frobenius norm
- `C = (1/d) Σ_{i≠j} C_ij²`
- Encourages independent features

### 5. Dynamic Weight Controller

Two strategies for balancing loss terms:

**Uncertainty Weighting** (default):
- Learn log-variances `s_I, s_V, s_C` as parameters
- Loss: `Σ_k [L_k·exp(-s_k) + s_k]`
- Automatic balancing, simple to implement

**Fixed Weighting**:
- Initialize: `λI:λV:λC = 9:0.5:0.5`
- Clamp within `[0.25×, 4×]` bounds

## Usage

### Training

```bash
# Enhanced Cosine Transformer with VIC regularization
python train.py \
    --method FSCT_enhanced_cosine \
    --dataset miniImagenet \
    --backbone ResNet18 \
    --n_way 5 \
    --k_shot 5 \
    --n_query 8 \
    --num_epoch 50 \
    --use_checkpoint 1 \
    --wandb 1
```

### Memory-Efficient Training (Kaggle Config)

```bash
python train.py \
    --method FSCT_enhanced_cosine \
    --dataset miniImagenet \
    --backbone ResNet18 \
    --n_way 5 \
    --k_shot 5 \
    --n_query 8 \
    --use_amp 1 \
    --use_checkpoint 1 \
    --grad_clip 1.0
```

### Testing

```bash
python test.py \
    --method FSCT_enhanced_cosine \
    --dataset miniImagenet \
    --backbone ResNet18 \
    --n_way 5 \
    --k_shot 5 \
    --test_iter 600
```

## New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use_amp` | int | 0 | Use automatic mixed precision (1) or not (0) |
| `--use_checkpoint` | int | 0 | Use gradient checkpointing (1) or not (0) |
| `--grad_clip` | float | 1.0 | Gradient clipping norm (0 to disable) |

## Methods Available

| Method | Description |
|--------|-------------|
| `FSCT_enhanced_cosine` | Enhanced transformer with cosine attention, VIC regularization, and Mahalanobis classifier |
| `FSCT_enhanced_softmax` | Enhanced transformer with softmax attention (baseline comparison) |
| `FSCT_cosine` | Original cosine transformer (without VIC or Mahalanobis) |
| `FSCT_softmax` | Original softmax transformer |

## Expected Performance Gains

Based on the problem statement:

1. **Cosine attention alone**: +5-20 points vs scaled dot-product
2. **VIC regularization**: +2-8 points (larger on 5-shot)
3. **Mahalanobis distance**: Additional improvement by respecting class covariance
4. **Combined system**: Up to 20+ points on weak baselines; smaller but meaningful gains on strong pre-trained backbones

## Implementation Details

### Files

- `methods/enhanced_transformer.py`: Main enhanced transformer implementation
- `methods/mahalanobis_classifier.py`: Mahalanobis distance classifier with shrinkage covariance
- `methods/vic_regularization.py`: VIC regularization and dynamic weight controller
- `test_components.py`: Unit tests for all components

### Key Features

- **Gradient Checkpointing**: Reduces memory by recomputing activations during backward pass
- **Mixed Precision**: Uses float16 for forward/backward, maintains float32 master weights
- **Shrinkage Covariance**: Stable covariance estimation with adaptive or fixed regularization
- **Cholesky Decomposition**: Efficient and stable matrix inversion

### Hyperparameters

Default configuration matches problem statement:

```python
EnhancedFewShotTransformer(
    depth=2,              # Transformer blocks
    heads=4,              # Attention heads
    dim_head=64,          # Dimension per head
    mlp_dim=512,          # FFN hidden dimension
    use_vic=True,         # Enable VIC regularization
    use_mahalanobis=True, # Enable Mahalanobis classifier
    vic_lambda_init=[9.0, 0.5, 0.5],  # Initial [λI, λV, λC]
    weight_controller='uncertainty',   # Dynamic weighting method
    use_checkpoint=False  # Enable for memory efficiency
)
```

## Testing

Run unit tests to verify components:

```bash
python test_components.py
```

This tests:
- Mahalanobis classifier forward pass and gradient
- VIC regularization computation
- Dynamic weight controller
- End-to-end integration

## Notes

1. **Image Size**: Use 84×84 for Conv/ResNet-12/18 on miniImageNet/CIFAR-FS
2. **Episode Size**: 5-way, k∈{1,5}, q=8 recommended
3. **Optimizer**: AdamW with lr=1e-3, weight_decay=1e-5
4. **Augmentation**: Mild augmentation; avoid aggressive aug for 1-shot
5. **Test Episodes**: Use 600 episodes for stable evaluation
