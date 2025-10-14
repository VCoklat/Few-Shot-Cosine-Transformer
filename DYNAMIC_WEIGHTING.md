# Dynamic Weighting Formula Implementation

This document describes the implementation of the dynamic weighting formula that combines three loss components to improve accuracy in few-shot learning.

## Overview

The implementation adds a dynamic weighting mechanism that combines:
1. **Cross-Entropy Loss (Equation 4)**: The standard classification loss
2. **Variance Regularization Term (Equation 5)**: Encourages stable variance across embedding dimensions
3. **Covariance Regularization Term (Equation 6)**: Penalizes correlations between dimensions

## Mathematical Formulation

### 1. Cross-Entropy Loss (Equation 4)
```
I = -log p_θ(y=k|Q)
```
Where:
- `I`: Invariance function (cross-entropy loss)
- `p_θ(y=k|Q)`: Probability of true class k for query Q

### 2. Variance Regularization Term (Equation 5)
```
V(E) = (1/m) * Σ_j max(0, γ - σ(E_j, ε))
```
Where:
- `E`: Concatenated support set embedding E_k and prototype embedding P_k
- `E_j`: Each dimension in E
- `σ(E_j, ε) = √(Var(E_j) + ε)`: Regularized standard deviation
- `γ`: Constant target value (fixed to 0.1 in experiments)
- `ε`: Small scalar preventing numerical instability (1e-8)
- `m`: Number of dimensions

### 3. Covariance Regularization Term (Equation 6)
```
C(E) = (1/(m-1)) * Σ_j (E_j - Ē)(E_j - Ē)^T
where Ē = (1/K) * Σ_i E_j
```
This computes the sum of squared off-diagonal coefficients of the covariance matrix.

Where:
- `Ē`: Mean of embeddings
- `K`: Number of classes

### Dynamic Weighting
The three loss components are combined using a learnable weight predictor:
```
Loss = w_ce * I + w_var * V(E) + w_cov * C(E)
```
Where `w_ce`, `w_var`, and `w_cov` are predicted by a neural network based on global embedding statistics, and constrained to sum to 1 using softmax.

## Implementation Details

### Classes Modified

#### 1. `FewShotTransformer` (methods/transformer.py)
- Added `variance_regularization()` method to compute V(E)
- Added `covariance_regularization()` method to compute C(E)
- Added `weight_predictor` neural network for dynamic weight prediction
- Modified `set_forward_loss()` to combine all three loss components

#### 2. `CTX` (methods/CTX.py)
- Same modifications as FewShotTransformer
- Adapted for the CrossTransformer architecture

### Parameters

Both classes now support the following additional parameters:

- `gamma` (float, default=0.1): Target value for variance regularization
- `epsilon` (float, default=1e-8): Small scalar for numerical stability
- `use_regularization` (bool, default=True): Enable/disable the regularization terms

### Weight Predictor Architecture

The weight predictor is a simple neural network:
```
Sequential(
    Linear(emb_dim * 2, emb_dim),    # Concatenate support and query global features
    LayerNorm(emb_dim),
    ReLU(),
    Linear(emb_dim, 3),              # Output 3 weights
    Softmax(dim=-1)                  # Ensure weights sum to 1
)
```

## Usage

### FewShotTransformer Example
```python
from methods.transformer import FewShotTransformer

# With regularization (default)
model = FewShotTransformer(
    model_func=ResNet10,
    n_way=5,
    k_shot=5,
    n_query=15,
    variant="softmax",
    gamma=0.1,              # Variance target
    epsilon=1e-8,           # Numerical stability
    use_regularization=True # Enable dynamic weighting
)

# Without regularization
model = FewShotTransformer(
    model_func=ResNet10,
    n_way=5,
    k_shot=5,
    n_query=15,
    use_regularization=False
)
```

### CTX Example
```python
from methods.CTX import CTX

# With regularization (default)
model = CTX(
    model_func=ResNet10,
    n_way=5,
    k_shot=5,
    n_query=15,
    variant="softmax",
    gamma=0.1,
    epsilon=1e-8,
    use_regularization=True
)
```

## Expected Benefits

1. **Improved Accuracy**: The dynamic weighting allows the model to adaptively balance the three objectives during training
2. **Better Generalization**: Variance and covariance regularization encourage more robust feature representations
3. **Stability**: The regularization terms help prevent overfitting and improve stability

## Implementation Notes

1. The regularization terms are computed on the concatenated support and query embeddings
2. The weight predictor uses global statistics (mean) of embeddings to predict weights per batch
3. Weights are normalized using softmax to ensure they sum to 1
4. The implementation is backward-compatible - setting `use_regularization=False` reverts to the original behavior

## References

This implementation is based on the mathematical formulations from the problem statement, specifically:
- Equation 4: Cross-Entropy Loss
- Equation 5: Variance Regularization
- Equation 6: Covariance Regularization
