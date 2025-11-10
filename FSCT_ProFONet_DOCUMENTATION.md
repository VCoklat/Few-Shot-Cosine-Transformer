# Hybrid FS-CT + ProFONet Algorithm Documentation

## Overview

This implementation combines the **Few-Shot Cosine Transformer (FS-CT)** with **ProFONet's VIC Regularization** to create a hybrid few-shot classification algorithm optimized for 8GB VRAM constraints.

## Algorithm Components

### 1. VIC Regularization Module

The VIC regularization prevents representation collapse through three complementary losses:

- **Variance Loss (V)**: Prevents norm collapse by ensuring embeddings maintain sufficient variance
  ```python
  V(E) = (1/m) * Σ max(0, γ - σ(E_j, ε))
  where σ(E_j, ε) = sqrt(Var(E_j) + ε)
  γ = 1.0, ε = 1e-6
  ```

- **Invariance Loss (I)**: Standard cross-entropy loss for classification
  ```python
  I = -log p_θ(y=k | Q)
  ```

- **Covariance Loss (C)**: Prevents feature correlation and redundancy
  ```python
  C(E) = (1/(m-1)) * Σ (E_j - Ē)(E_j - Ē)^T
  C_loss = Σ(off_diagonal(C(E))^2) / m
  ```

### 2. Dynamic Weight Scheduler

Adjusts regularization weights based on training progress:

```python
epoch_ratio = current_epoch / total_epochs

λ_V = 0.5 * (1 + 0.3 * epoch_ratio)  # Increases over time
λ_I = 9.0                             # Constant (dominant)
λ_C = 0.5 * (1 - 0.2 * epoch_ratio)  # Decreases over time
```

**Rationale**: Early training focuses on decorrelating features (high C), while later training emphasizes maintaining variance (high V).

### 3. Learnable Prototypical Embedding

Computes class prototypes using learnable weights:

```python
W_avg = softmax(θ_P)  # Shape: (n, k, 1)
ZP = Σ(ZS ⊙ W_avg, axis=k)  # Shape: (n, d)
```

This allows the model to learn which support samples are most representative.

### 4. Cosine Attention Transformer

Multi-head attention using cosine similarity (no softmax):

```python
# Compute cosine similarity
qk = q @ k.transpose(-2, -1)
M_q = ||q||_2
M_k = ||k||_2
A = qk ⊘ (M_q ⊗ M_k)  # A ∈ [-1, 1]

# Element-wise multiplication (no softmax!)
h_a = A ⊙ v
```

**Benefits**:
- Bounded attention weights without softmax
- More stable gradients
- Better correlation measurement

### 5. Combined Loss Function

```python
L_total = λ_V * V(E) + λ_I * I + λ_C * C(E)
```

With gradient clipping (max_norm=1.0) for stability.

## Memory Optimization Features

### For 8GB VRAM Constraint

1. **Reduced Configuration**:
   - Attention heads: 4 (instead of 8)
   - Head dimension: 160 (instead of 80)
   - Query samples: 10 (instead of 16)
   - Embedding dimension: 1600 (Conv4) or 640 (for larger backbones)

2. **Gradient Checkpointing** (enabled on CUDA):
   ```python
   x = torch.utils.checkpoint.checkpoint(self.ATTN, x, query, query) + x
   x = torch.utils.checkpoint.checkpoint(self.FFN, x) + x
   ```

3. **Mixed Precision Training** (optional, enabled on CUDA):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       loss = compute_loss()
   ```

4. **Efficient Backbones**:
   - Recommended: Conv4, Conv6, ResNet12
   - Input size: 84×84 (instead of 224×224)

## Usage

### Training Example

```bash
python train.py \
  --method FSCT_ProFONet \
  --dataset miniImagenet \
  --backbone Conv4 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 10 \
  --num_epoch 50 \
  --learning_rate 0.001 \
  --optimization AdamW \
  --weight_decay 1e-5
```

### Testing Example

```bash
python test.py \
  --method FSCT_ProFONet \
  --dataset miniImagenet \
  --backbone Conv4 \
  --n_way 5 \
  --k_shot 5
```

## Configuration Options

### Model Parameters

- `variant`: 'cosine' or 'softmax' (default: 'cosine')
- `depth`: Number of transformer layers (default: 1)
- `heads`: Number of attention heads (default: 4)
- `dim_head`: Dimension per attention head (default: 160)
- `mlp_dim`: FFN hidden dimension (default: 512)
- `dropout`: Dropout rate (default: 0.0)

### VIC Regularization Parameters

- `lambda_V_base`: Base variance weight (default: 0.5)
- `lambda_I`: Invariance weight (default: 9.0)
- `lambda_C_base`: Base covariance weight (default: 0.5)

### Memory Optimization

- `gradient_checkpointing`: Enable gradient checkpointing (default: True on CUDA)
- `mixed_precision`: Enable mixed precision training (default: True on CUDA)

## Architecture Details

### Model Flow

```
Input Images (n, k+q, 3, H, W)
    ↓
Feature Extractor (backbone)
    ↓
Support Features (n, k, d)
    ↓
Learnable Weighted Prototypes (n, d)
    ↓
Cosine Attention Transformer
    ↓
Cosine Linear Layer
    ↓
Classification Scores (q, n)
```

### VIC Regularization Application

```
Support Features + Prototypes → Embeddings (n*k + n, d)
    ↓
VIC Regularization
    ├─ Variance Loss
    ├─ Covariance Loss
    └─ Combined with Invariance Loss
```

## Expected Performance

Based on the algorithm design:

- **Target Improvement**: >20% accuracy improvement over baseline
- **Memory Usage**: Optimized for 8GB VRAM
- **Training Stability**: Enhanced by gradient clipping and dynamic weighting

### Theoretical Advantages

1. **VIC Regularization**: Prevents representation collapse
2. **Dynamic Weights**: Adaptive regularization during training
3. **Cosine Attention**: More stable than softmax attention
4. **Learnable Prototypes**: Better class representation

## Testing

### Run Unit Tests

```bash
python test_fsct_profonet.py
```

Tests cover:
- VIC Regularization Module
- Dynamic Weight Scheduler
- Cosine Attention Layer
- Model initialization
- Forward pass
- Loss computation
- Epoch setting

### Run Integration Tests

```bash
python test_integration.py
```

Tests cover:
- Method selection
- Model instantiation with different backbones
- Training step
- Validation step
- Memory optimization features

## Implementation Notes

### Key Differences from Standard FS-CT

1. **Added VIC Regularization**: Three complementary losses
2. **Dynamic Weight Adjustment**: Weights change during training
3. **Gradient Clipping**: Added for stability
4. **Memory Optimizations**: Gradient checkpointing and mixed precision
5. **Reduced Configuration**: Optimized for 8GB VRAM

### Key Differences from ProFONet

1. **Cosine Attention**: Uses cosine similarity instead of dot-product
2. **No Softmax in Attention**: Direct cosine similarity
3. **Learnable Prototypes**: Weighted averaging instead of simple mean
4. **Transformer Architecture**: Multi-head attention with skip connections

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `n_query` (try 8 instead of 10)
   - Enable gradient checkpointing
   - Use smaller backbone (Conv4 instead of ResNet)

2. **Training Instability**:
   - Check gradient clipping is enabled
   - Verify dynamic weight scheduling is working
   - Monitor loss components (V, I, C)

3. **Poor Performance**:
   - Verify VIC regularization weights are reasonable
   - Check if variance loss is not collapsing
   - Monitor covariance loss trend

## References

1. **FS-CT**: "Enhancing Few-shot Image Classification with Cosine Transformer" (IEEE Access 2023)
2. **ProFONet**: VIC Regularization for few-shot learning
3. **VICReg**: "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" (ICLR 2022)

## Citation

If you use this implementation, please cite:

```bibtex
@article{nguyen2023FSCT,
  author={Nguyen, Quang-Huy and Nguyen, Cuong Q. and Le, Dung D. and Pham, Hieu H.},
  journal={IEEE Access}, 
  title={Enhancing Few-Shot Image Classification With Cosine Transformer}, 
  year={2023},
  volume={11},
  pages={79659-79672},
  doi={10.1109/ACCESS.2023.3298299}
}
```

## License

This implementation follows the license of the original Few-Shot-Cosine-Transformer repository.
