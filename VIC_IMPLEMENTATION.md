# VIC-Enhanced FS-CT Implementation Summary

## Overview
This implementation adds VIC (Variance-Invariance-Covariance) loss components to the Few-Shot Cosine Transformer (FS-CT) training process, as specified in the algorithm for "Episodic Training for VIC-Enhanced FS-CT".

## Changes Made

### 1. Added VIC Loss Parameters (`io_utils.py`)
- `--lambda_I` (default: 1.0): Weight for Invariance Loss (standard cross-entropy)
- `--lambda_V` (default: 0.0): Weight for Variance Loss
- `--lambda_C` (default: 0.0): Weight for Covariance Loss

### 2. Implemented VIC Loss Functions (`methods/transformer.py`)

#### FewShotTransformer class modifications:
- Added lambda parameters to `__init__`: `lambda_I`, `lambda_V`, `lambda_C`
- Added `variance_loss()` method: Implements hinge loss on standard deviation of support embeddings
- Added `covariance_loss()` method: Implements covariance regularization to decorrelate features
- Modified `set_forward_loss()`: Computes combined loss L_total = (λ_I × L_I) + (λ_V × L_V) + (λ_C × L_C)

### 3. Updated Training Scripts
- `train.py`: Pass VIC loss weights to FewShotTransformer
- `train_test.py`: Pass VIC loss weights to FewShotTransformer

### 4. Updated Documentation
- `README.md`: Added VIC loss parameter documentation and usage examples

## Algorithm Implementation

### Combined Loss Formula:
```
L_total = (λ_I × L_I) + (λ_V × L_V) + (λ_C × L_C)
```

### Loss Components:

1. **Invariance Loss (L_I)**:
   - Standard Categorical Cross-Entropy Loss
   - Ensures predictions match ground truth labels
   - Implementation: Uses existing `nn.CrossEntropyLoss()`

2. **Variance Loss (L_V)**:
   - Hinge loss: `max(0, γ - std(Z_S))`
   - Encourages compact support set embeddings per class
   - Implementation:
     ```python
     std_per_class = torch.std(z_support, dim=1)
     hinge_loss = F.relu(gamma - std_per_class)
     loss_v = hinge_loss.mean()
     ```

3. **Covariance Loss (L_C)**:
   - Minimizes off-diagonal elements of covariance matrix
   - Decorrelates feature dimensions to prevent collapse
   - Implementation:
     ```python
     z_centered = z_flat - z_flat.mean(dim=0)
     cov_matrix = (z_centered.T @ z_centered) / (batch_size - 1)
     loss_c = sum of squared off-diagonal elements
     ```

## Usage Examples

### Standard Training (Original Behavior):
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet \
    --backbone ResNet34 --n_way 5 --k_shot 5
```

### VIC-Enhanced Training:
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet \
    --backbone ResNet34 --n_way 5 --k_shot 5 \
    --lambda_I 1.0 --lambda_V 0.5 --lambda_C 0.1
```

### Only Variance Loss:
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet \
    --backbone ResNet34 --n_way 5 --k_shot 5 \
    --lambda_I 1.0 --lambda_V 1.0 --lambda_C 0.0
```

## Backward Compatibility
- Default values (`lambda_V=0.0`, `lambda_C=0.0`) maintain original training behavior
- No breaking changes to existing code
- All existing training scripts work without modification
- VIC losses are only computed when weights > 0

## Testing
All tests pass successfully:

1. **Unit Tests** (`test_vic_loss.py`):
   - Model initialization with default/custom weights ✓
   - Variance loss computation ✓
   - Covariance loss computation ✓
   - Forward/backward pass ✓
   - Gradient computation ✓

2. **Integration Tests** (`test_vic_integration.py`):
   - Complete episodic training step ✓
   - Multiple training episodes ✓
   - Different VIC weight configurations ✓
   - Loss component verification ✓

## Technical Details

### Tensor Shapes:
- Support embeddings: `(n_way, k_shot, feat_dim)`
- Query embeddings: `(n_way * n_query, feat_dim)`
- Variance loss operates per class on support set
- Covariance loss operates on flattened support embeddings

### Performance Considerations:
- VIC losses are only computed when weights > 0
- Minimal overhead when disabled (default)
- No impact on inference/testing

### Gradient Flow:
- All loss components are differentiable
- Gradients flow through entire model
- Standard backpropagation works correctly

## References
- VICReg: Variance-Invariance-Covariance Regularization (Bardes et al., 2022)
- ProFONet: Prototypical Few-shot Object Detection Network (Xiao et al., 2022)
- Few-Shot Cosine Transformer (Nguyen et al., 2023)
