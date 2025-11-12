# Weight Prediction Stabilization

This document describes the improvements made to stabilize the weight prediction mechanism in the Few-Shot Cosine Transformer. These changes implement the highest ROI (Return on Investment) improvements from the problem statement.

## Overview

The weight predictor dynamically combines three attention components (cosine similarity, variance, and covariance) to improve few-shot learning accuracy. However, this can be unstable during training. We've implemented several key stabilization techniques to address this.

## Key Improvements

### 1. Temperature Parameter for Softmax

**Implementation:** `methods/transformer.py` - `Attention.__init__()` and `weight_predictor_forward()`

The weight predictor now uses a learnable temperature parameter for the softmax that produces the 3 component weights:

```python
self.weight_temperature = nn.Parameter(torch.ones(heads) * 1.0)
```

**Benefits:**
- **Lower temperature** (< 1.0) → crisper choices, more confident decisions
- **Higher temperature** (> 1.0) → smoother mixing, more exploration
- **Per-head temperature** allows each attention head to learn its own mixing strategy

**Formula:**
```python
weights = F.softmax(logits / temperature, dim=-1)
```

### 2. Entropy Regularization

**Implementation:** `methods/transformer.py` - `weight_predictor_forward()` and `forward()`

We add an entropy regularizer to encourage moderate entropy in predicted weights:

```python
self.entropy_reg_lambda = 0.01  # Configurable penalty weight
target_entropy = np.log(3.0)  # Target for 3 components ≈ 1.1
entropy_reg = self.entropy_reg_lambda * torch.mean((entropy - target_entropy) ** 2)
```

**Benefits:**
- Prevents collapse to a single component too early
- Encourages balanced use of all three components
- Moderate entropy (neither too peaked nor too uniform) leads to better generalization

### 3. L2 Penalty on Logit Magnitudes

**Implementation:** `methods/transformer.py` - `forward()`

We penalize large logit magnitudes to prevent extreme predictions:

```python
self.logit_l2_lambda = 0.001  # Configurable penalty weight
logit_l2 = self.logit_l2_lambda * torch.mean(logits ** 2)
```

**Benefits:**
- Prevents weight predictor from becoming overconfident
- Improves training stability
- Reduces risk of numerical issues

### 4. Gradient Clipping with Separate Learning Rate

**Implementation:** `train.py` - `train()` function

The weight predictor uses a smaller learning rate (0.5x main LR) for more stable training:

```python
param_groups = [
    {'params': other_params, 'lr': learning_rate},
    {'params': weight_predictor_params, 'lr': learning_rate * 0.5}
]
```

**Benefits:**
- Weight predictor learns more slowly, reducing instability
- Main model can still learn quickly
- Gradient clipping (max_norm=1.0) prevents explosion

### 5. Shrinkage Covariance Estimation

**Implementation:** `methods/transformer.py` - `covariance_component_torch()`

We use Ledoit-Wolf style shrinkage to reduce noise in covariance estimation:

```python
self.shrinkage_alpha = 0.1  # Balance parameter
cov_shrunk = (1 - alpha) * empirical_cov + alpha * diag(empirical_cov)
```

**Benefits:**
- Reduces noise in small mini-batches
- More stable covariance estimates
- Better generalization to unseen data

**Formula:**
```
Σ_shrunk = (1 - α) * Σ_empirical + α * diag(Σ_empirical)
```

### 6. Improved Numerical Stability

**Implementation:** `methods/transformer.py` - `variance_component_torch()` and `covariance_component_torch()`

We use stable numeric operations to prevent numerical issues:

```python
# Clamp before sqrt to avoid negative values
variance_per_dim = torch.clamp(variance_per_dim, min=epsilon)
# Safe sqrt with larger epsilon
regularized_std = torch.sqrt(variance_per_dim + epsilon)
```

**Benefits:**
- Prevents NaN and Inf values
- Handles extreme values gracefully
- More robust training

### 7. Component Magnitude Normalization

**Implementation:** `methods/transformer.py` - `forward()`

We normalize components to similar dynamic range before mixing:

```python
# Normalize by running std to ensure similar scale
cosine_std = cosine_sim.std() + epsilon
var_std = var_component.std() + epsilon
cov_std = cov_component.std() + epsilon

cosine_sim_norm = cosine_sim / cosine_std
var_component_norm = var_component / (var_ema + epsilon) / var_std
cov_component_norm = cov_component / (cov_ema + epsilon) / cov_std
```

**Benefits:**
- All components have similar magnitude
- Softmax can effectively learn to balance them
- Prevents one component from dominating simply due to scale

### 8. Increased Dropout in Weight Predictor

**Implementation:** `methods/transformer.py` - `Attention.__init__()`

We use more dropout to prevent overfitting to noise:

```python
self.weight_dropout1 = nn.Dropout(0.15)  # Increased from 0.1
self.weight_dropout2 = nn.Dropout(0.1)   # Additional layer
```

**Benefits:**
- Prevents overfitting to noise in qk_features
- Improves generalization
- More robust predictions

## Usage

### Basic Usage

The improvements are automatically enabled when using `dynamic_weight=True`:

```python
from methods.transformer import FewShotTransformer

model = FewShotTransformer(
    model_func=backbone_func,
    n_way=5,
    k_shot=5,
    n_query=15,
    variant="cosine",
    dynamic_weight=True,  # Enable dynamic weighting with stabilization
    heads=8,
    dim_head=64
)
```

### Advanced Configuration

You can customize the stabilization parameters:

```python
# Access the attention module
attention = model.ATTN

# Adjust temperature (lower = crisper, higher = smoother)
attention.weight_temperature.data = torch.ones(heads) * 0.5

# Adjust entropy regularization weight
attention.entropy_reg_lambda = 0.02  # Higher penalty on entropy deviation

# Adjust L2 penalty on logits
attention.logit_l2_lambda = 0.002  # Higher penalty on large logits

# Adjust shrinkage coefficient
attention.shrinkage_alpha = 0.2  # More shrinkage (0-1 range)
```

### Training with Separate Learning Rates

The training script automatically detects the weight predictor and uses a separate learning rate:

```python
# In train.py, this is automatically handled
python train.py --method transformer --dynamic_weight
```

The weight predictor will use 0.5x the main learning rate for more stable training.

## Monitoring and Debugging

### Track Weight Statistics

Enable weight recording to track the weight distribution:

```python
model.ATTN.record_weights = True
# ... run evaluation ...
stats = model.ATTN.get_weight_stats()
print(stats)
# Output: {'cosine_mean': 0.4, 'cov_mean': 0.3, 'var_mean': 0.3, ...}
```

### Check Regularization Losses

Access the regularization losses during training:

```python
# After forward pass
entropy_reg = model.ATTN.last_entropy_reg
logit_l2 = model.ATTN.last_logit_l2
total_reg = model.ATTN.get_regularization_losses()
```

### Monitor Component Magnitudes

Track the EMA values to see component scales:

```python
print(f"Variance EMA: {model.ATTN.var_ema.item()}")
print(f"Covariance EMA: {model.ATTN.cov_ema.item()}")
```

## Testing

Run the comprehensive test suite to validate the implementation:

```bash
python test_weight_stabilization.py
```

This tests:
- Temperature parameter initialization and usage
- Entropy regularization computation
- L2 penalty on logits
- Shrinkage covariance estimation
- Numerical stability with extreme values
- Component normalization
- Dropout in weight predictor
- Full integration with the model

## Performance Impact

Based on the problem statement, these improvements should provide:

1. **More stable training**: Reduced variance in training curves
2. **Better convergence**: Faster convergence to good solutions
3. **Improved accuracy**: 2-5% improvement expected
4. **Reduced risk of collapse**: Weight predictor less likely to collapse to single component
5. **Better generalization**: More robust to distribution shift

## References

- **Problem Statement**: Stabilize weight prediction (highest ROI)
- **Ledoit-Wolf Shrinkage**: Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices"
- **Entropy Regularization**: Pereyra et al. (2017). "Regularizing neural networks by penalizing confident output distributions"
- **Temperature Scaling**: Hinton et al. (2015). "Distilling the knowledge in a neural network"

## Future Improvements

Potential additional improvements not yet implemented:

1. **Curriculum learning**: Start with cosine-only, gradually enable other components
2. **Progressive unfreezing**: Freeze weight predictor for first N epochs
3. **Auxiliary loss**: Encourage predictor to choose best component via ablation
4. **Temporal smoothing**: Penalize rapid weight changes across steps
5. **EMA-based covariance**: Use running buffer instead of batch-only
6. **Per-head features**: Give predictor access to head-specific aggregated features
7. **Learnable scale factors**: Per-component multiplicative scaling
8. **Correlation penalty**: Use normalized correlation instead of raw covariance
