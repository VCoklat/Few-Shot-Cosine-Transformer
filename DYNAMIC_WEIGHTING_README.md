# Dynamic Weighting Feature

This repository now includes a dynamic weighting mechanism that combines three complementary formulas to improve few-shot learning accuracy:

## Quick Example

```python
from methods.transformer import FewShotTransformer

# Create model with dynamic weighting
model = FewShotTransformer(
    model_func=backbone,
    n_way=5,
    k_shot=5,
    n_query=15,
    variant="cosine",
    dynamic_weight=True  # Enable dynamic weighting
)

# Train as usual - dynamic weighting works automatically
for batch in train_loader:
    acc, loss = model.set_forward_loss(batch)
    loss.backward()
    optimizer.step()
```

## Features

✓ **Invariance** - Cosine similarity for semantic matching  
✓ **Variance Regularization** - Prevents feature collapse  
✓ **Covariance Regularization** - Reduces feature redundancy  
✓ **Dynamic Weighting** - Learns optimal combination  
✓ **OOM Prevention** - Chunked processing for large models  

## Documentation

- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Complete overview
- [DYNAMIC_WEIGHTING.md](DYNAMIC_WEIGHTING.md) - Technical details
- [USAGE.md](USAGE.md) - Usage guide

## Validation

```bash
python validate_formulas.py
```

This validates all three formulas work correctly together.
