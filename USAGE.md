# Using Dynamic Weighting in Few-Shot Cosine Transformer

## Overview

The dynamic weighting feature combines three complementary formulas to improve few-shot learning accuracy:

1. **Invariance** - Cosine similarity for semantic matching
2. **Variance Regularization** - Encourages feature diversity
3. **Covariance Regularization** - Reduces feature redundancy

## Quick Start

### Basic Usage

```python
from methods.transformer import FewShotTransformer

# Create model with dynamic weighting enabled
model = FewShotTransformer(
    model_func=backbone_model,
    n_way=5,
    k_shot=5,
    n_query=15,
    variant="cosine",
    dynamic_weight=True,  # Enable dynamic weighting
    initial_cov_weight=0.3,
    initial_var_weight=0.5
)
```

### Training with Dynamic Weighting

```python
# The model will automatically use dynamic weighting during training
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        acc, loss = model.set_forward_loss(batch)
        loss.backward()
        optimizer.step()
```

### Advanced Usage

You can control the advanced attention mechanism:

```python
# Enable advanced attention components
model.use_advanced_attention = True

# Set gamma parameter for variance regularization
model.gamma = 1.0  # Default value

# Set epsilon for numerical stability
model.epsilon = 1e-8  # Default value
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dynamic_weight` | `False` | Enable dynamic weight prediction |
| `initial_cov_weight` | `0.3` | Initial weight for covariance component (if not dynamic) |
| `initial_var_weight` | `0.5` | Initial weight for variance component (if not dynamic) |
| `gamma` | `1.0` | Hinge threshold for variance regularization |
| `epsilon` | `1e-8` | Small constant for numerical stability |

## Expected Improvements

### Accuracy
- Better feature representations through variance regularization
- Reduced redundancy via covariance regularization
- Improved semantic matching with cosine similarity
- Adaptive weighting learns optimal combination for each task

### Memory Efficiency
- Chunked processing prevents OOM errors
- Explicit memory management with `torch.cuda.empty_cache()`
- Adaptive chunk sizes based on available memory

## Validation

Run the validation script to verify the implementation:

```bash
python validate_formulas.py
```

This will test all three formulas and confirm they work correctly together.

## Technical Details

For detailed information about the formulas and implementation, see:
- [DYNAMIC_WEIGHTING.md](DYNAMIC_WEIGHTING.md) - Complete technical documentation

## Troubleshooting

### Out of Memory Errors
If you encounter OOM errors:
1. Reduce batch size
2. Reduce number of attention heads
3. Reduce feature dimension

The implementation includes OOM prevention mechanisms, but very large models may still require adjustment.

### Weight Analysis
To analyze the learned weights during evaluation:

```python
from methods.transformer import Attention

# Enable weight recording
for module in model.modules():
    if isinstance(module, Attention):
        module.record_weights = True

# Run evaluation
model.eval()
with torch.no_grad():
    for batch in val_loader:
        model.set_forward(batch)

# Get weight statistics
for module in model.modules():
    if isinstance(module, Attention):
        stats = module.get_weight_stats()
        if stats:
            print(f"Cosine weight: {stats['cosine_mean']:.3f}")
            print(f"Covariance weight: {stats['cov_mean']:.3f}")
            print(f"Variance weight: {stats['var_mean']:.3f}")
```

## References

The formulas are based on:
- Invariance: Cross-entropy loss with softmax probabilities
- Variance Regularization: Multi-dimensional variance with hinge loss
- Covariance Regularization: Off-diagonal covariance matrix penalties

See the problem statement for the original mathematical formulations.
