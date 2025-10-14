# Dynamic Weighting Formula Implementation

This implementation combines three key formulas to improve few-shot learning accuracy:

## 1. Invariance Component (Cosine Similarity)

**Formula from problem statement:**
```python
def invariance(logits, true_class):
    probabilities = softmax(logits)
    p_true_class = probabilities[true_class]
    loss = -np.log(p_true_class)
    return loss
```

**Implementation:**
- Implemented via cosine similarity in the attention mechanism
- Computes cosine distance between query and key features
- Located in `cosine_distance()` function and used in attention forward pass

## 2. Variance Regularization

**Formula from problem statement:**
```python
def variance_regularization_multi_dim(E, gamma=0.1, epsilon=1e-8):
    variance_per_dim = np.var(E, axis=0, ddof=0)
    regularized_std = np.sqrt(variance_per_dim + epsilon)
    hinge_values = np.maximum(0.0, gamma - regularized_std)
    V_E = np.sum(hinge_values) / m
    return V_E
```

**Implementation in `variance_component_torch()`:**
```python
# Reshape to compute variance across all samples
E_reshaped = E.reshape(-1, E.shape[-1])  # (batch*seq, dim)

# Compute variance per dimension across samples (axis=0)
variance_per_dim = torch.var(E_reshaped, dim=0, unbiased=False)

# Compute regularized standard deviation
regularized_std = torch.sqrt(variance_per_dim + epsilon)

# Apply hinge: max(0, gamma - regularized_std)
hinge_values = torch.clamp(gamma - regularized_std, min=0.0)

# Sum and normalize by number of samples
V_E = torch.sum(hinge_values) / E_reshaped.shape[0]
```

**Purpose:**
- Encourages variance in feature representations
- Prevents feature collapse where all features become similar
- Uses hinge loss to penalize low variance

## 3. Covariance Regularization

**Formula from problem statement:**
```python
def covariance_regularization(E):
    E_mean = np.mean(E, axis=0, keepdims=True)
    E_centered = E - E_mean
    cov_matrix = np.dot(E_centered.T, E_centered) / (K - 1)
    mask = np.ones_like(cov_matrix) - np.eye(cov_matrix.shape[0])
    off_diagonal_squared = np.sum((cov_matrix * mask) ** 2)
    return off_diagonal_squared
```

**Implementation in `covariance_component_torch()`:**
```python
# Reshape to compute covariance across all samples
E_reshaped = E.reshape(-1, dim)

# Compute mean and center the data
E_mean = torch.mean(E_reshaped, dim=0, keepdim=True)
E_centered = E_reshaped - E_mean

# Compute covariance matrix with chunking (OOM prevention)
cov_matrix = torch.matmul(E_centered.T, E_centered) / (K - 1)

# Mask off-diagonal elements
mask = torch.ones_like(cov_matrix) - torch.eye(dim)

# Sum of squares of off-diagonal elements
off_diagonal_squared = torch.sum((cov_matrix * mask) ** 2)
```

**Purpose:**
- Reduces correlation between different features
- Encourages feature independence
- Penalizes off-diagonal covariance matrix elements

## Dynamic Weight Combination

The three components are combined using learned weights:

```python
# Dynamic weight prediction based on query-key features
weights = weight_predictor_forward(qk_features)  # [heads, 3]

cos_weight = weights[:, 0]  # Cosine weight
cov_weight = weights[:, 1]  # Covariance weight
var_weight = weights[:, 2]  # Variance weight

# Combine all three components
dots = (cos_weight * cosine_sim +
        cov_weight * cov_component + 
        var_weight * var_component)
```

**Weight Prediction:**
- Uses a small neural network to predict optimal weights
- Weights sum to 1.0 via softmax
- Adapts weights dynamically based on input features

## Memory Optimization (OOM Prevention)

Several strategies prevent out-of-memory errors:

1. **Chunked Processing:**
   - Covariance matrix computed in chunks
   - Large tensors processed in smaller batches
   - Explicit memory clearing with `torch.cuda.empty_cache()`

2. **Efficient Matrix Operations:**
   - Avoids materializing large intermediate tensors
   - Uses in-place operations where possible
   - Chunk size adapts based on available memory

3. **Error Handling:**
   - Fallback to basic attention if advanced fails
   - Graceful degradation when OOM occurs
   - Warning messages for debugging

## Usage

The implementation is used in the `Attention` class:

```python
# Create attention with dynamic weighting
attention = Attention(
    dim=512,
    heads=8,
    dim_head=64,
    variant="cosine",
    dynamic_weight=True  # Enable dynamic weighting
)

# Forward pass with dynamic weighting
output = attention(q, k, v, use_advanced=True, gamma=1.0, epsilon=1e-8)
```

## Expected Benefits

1. **Increased Accuracy:**
   - Combines complementary similarity measures
   - Learns optimal weighting for each task
   - Regularizes feature representations

2. **Better Generalization:**
   - Variance regularization prevents overfitting
   - Covariance regularization reduces redundancy
   - Invariance captures semantic similarity

3. **Memory Efficiency:**
   - Chunked processing handles large batches
   - Explicit memory management prevents OOM
   - Scalable to various model sizes

## Testing

Run the test script to verify the implementation:

```bash
python /tmp/test_components.py
```

This validates that all three formulas are correctly implemented.
