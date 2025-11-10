# Enhanced Few-Shot Cosine Transformer (EnhancedFSCT)

## Overview

EnhancedFSCT is an advanced few-shot learning method that combines:
1. **Learnable Weighted Prototypes**: Dynamically learns optimal shot weights per class
2. **Cosine Cross-Attention**: Multi-head attention using cosine similarity (no softmax)
3. **Mahalanobis Distance Classifier**: Uses class-specific covariance for robust classification
4. **VIC Regularization**: Variance-Invariance-Covariance losses for better feature learning
5. **Dynamic Loss Weighting**: Automatically balances multiple loss terms

This implementation follows specifications from:
- **FS-CT**: Enhancing Few-Shot Image Classification With Cosine Transformer
- **ProFONet**: VIC regularization for few-shot learning
- **VICReg**: Variance-Invariance-Covariance Regularization
- **GradNorm**: Gradient Normalization for Adaptive Loss Balancing
- **Mahalanobis-FSL**: Mahalanobis distance for few-shot learning

## Architecture Components

### 1. Learnable Weighted Prototypes

Instead of simple averaging, each class learns optimal weights for its support shots:

```
weights_c = softmax(learnable_params[c])  # Per-class learnable weights
prototype_c = Σ(weights_c[i] * support_c[i])  # Weighted combination
```

- **Initialization**: Uniform weights (zeros in log-space)
- **Training**: Weights adapt episodically to emphasize informative shots
- **Benefits**: Robust to outliers, adapts to class-specific patterns

### 2. Cosine Cross-Attention Encoder

Multi-head cross-attention using cosine similarity instead of scaled dot-product:

**Architecture**:
- 2 encoder blocks (configurable depth)
- 4 heads with 64 dimensions per head (H=4, dh=64)
- GELU activation in FFN
- Pre-norm LayerNorm
- Residual connections

**Attention Mechanism**:
```
Q = queries from query embeddings
K, V = keys/values from prototype embeddings
Q_norm = normalize(Q, dim=-1)
K_norm = normalize(K, dim=-1)
Attn = Q_norm @ K_norm^T  # Cosine similarity (no softmax!)
Output = Attn @ V
```

**Key Differences from Standard Attention**:
- No softmax normalization
- Uses L2-normalized queries and keys
- More stable attention patterns for few-shot scenarios

### 3. Mahalanobis Distance Classifier

Replaces Euclidean distance with Mahalanobis distance that respects class covariance:

```
For each class c:
  Σ_c = (1-α)·Cov(support_c) + α·I  # Shrinkage covariance
  D_c(q) = (q - proto_c)^T · Σ_c^(-1) · (q - proto_c)
  
scores = -D  # Negative distances as logits
```

**Shrinkage Parameter**:
- Adaptive: α = d/(k+d) where d=dimensions, k=shots
- Or fixed: α ∈ [0.1, 0.2]
- Regularizes covariance for numerical stability

**Benefits**:
- Accounts for class-specific feature variance
- Robust to anisotropic feature distributions
- Better separation in embedding space

### 4. VIC Regularization

Three complementary losses for better feature learning:

#### Variance Loss (V)
Encourages high variance per dimension (σ ≈ 1):
```
σ_j = std(embeddings[:, j])
V = mean(max(0, σ_j - 1 - ε))  # Hinge loss
```

#### Invariance Loss (I)
Standard classification loss via Mahalanobis distances:
```
I = CrossEntropy(logits=-distances, targets)
```

#### Covariance Loss (C)
Decorrelates features (encourages diagonal covariance):
```
Cov = covariance_matrix(normalized_embeddings)
C = mean((off_diagonal(Cov))^2)
```

**Total Loss**:
```
L = λ_I · I + λ_V · V + λ_C · C
```

### 5. Dynamic Loss Weighting

Three strategies for balancing VIC losses:

#### Uncertainty Weighting (Default)
Learns log-variances for automatic balancing:
```
L_total = Σ_k [L_k · exp(-s_k) + s_k]
where k ∈ {I, V, C} and s_k are learnable parameters
```

**Advantages**:
- No manual tuning needed
- Automatically balances loss scales
- Simple and stable

#### GradNorm Controller
Adjusts weights based on gradient norms:
```
Target gradient norm for loss k: Ḡ · (r_k)^α
where r_k is relative loss rate, α ∈ [0.5, 1.5]
```

**Advantages**:
- Explicit gradient balancing
- Configurable adaptation rate
- Good for complex multi-task scenarios

#### Stats-Driven Fallback
Weights based on current loss statistics:
```
λ_V ∝ |mean(σ_j) - 1|
λ_C ∝ off_diagonal_norm(Cov)
λ_I ∝ moving_average(CE)
```

**Advantages**:
- No extra parameters
- Interpretable weights
- Good for debugging

## Training Configuration

### Recommended Hyperparameters

```python
# Architecture
depth = 2              # Encoder blocks
heads = 4              # Attention heads
dim_head = 64          # Dimensions per head
mlp_dim = 512          # FFN hidden dimension

# VIC Loss Weights (initial)
lambda_I = 9.0         # Classification (Invariance)
lambda_V = 0.5         # Variance
lambda_C = 0.5         # Covariance

# Dynamic Weighting
use_uncertainty_weighting = True  # Recommended
use_gradnorm = False              # Alternative
shrinkage_alpha = None            # Adaptive: d/(k+d)

# Training
optimizer = 'AdamW'
learning_rate = 1e-3
weight_decay = 1e-5
num_epochs = 50
episodes_per_epoch = 200

# Episode Configuration
n_way = 5
k_shot = 1 or 5
n_query = 8            # Memory-efficient

# Mixed Precision (optional)
use_amp = True         # Enable for memory savings
grad_clip = 1.0        # Gradient clipping norm
```

### Image Sizes

- **84×84**: Conv4/Conv6, ResNet-12/18 on CIFAR-FS/mini-ImageNet
- **112×112**: For stronger backbones or larger datasets

### Memory Optimization (8GB VRAM)

1. **Mixed Precision**: `use_amp=True`
2. **Gradient Checkpointing**: Automatic in encoder blocks
3. **Efficient Episode Sizes**: q=8 queries per class
4. **Shrinkage Covariance**: Reduces memory vs. full covariance
5. **Gradient Clipping**: Prevents gradient explosion

## Usage Examples

### Training with EnhancedFSCT

```bash
# 5-way 1-shot on mini-ImageNet with ResNet18
python train_test.py \
  --method EnhancedFSCT \
  --dataset miniImagenet \
  --backbone ResNet18 \
  --n_way 5 \
  --k_shot 1 \
  --n_query 8 \
  --num_epoch 50 \
  --learning_rate 1e-3 \
  --optimization AdamW \
  --use_amp 1 \
  --grad_clip 1.0

# 5-way 5-shot on CUB with ResNet34 and uncertainty weighting
python train_test.py \
  --method EnhancedFSCT \
  --dataset CUB \
  --backbone ResNet34 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 8 \
  --use_uncertainty 1 \
  --lambda_I 9.0 \
  --lambda_V 0.5 \
  --lambda_C 0.5 \
  --use_amp 1
```

### Programmatic Usage

```python
from methods.enhanced_fsct import EnhancedFSCT
import backbone

# Define feature extractor
def feature_model():
    return backbone.ResNet18(FETI=0, dataset='miniImagenet', flatten=True)

# Create model
model = EnhancedFSCT(
    feature_model,
    n_way=5,
    k_shot=1,
    n_query=8,
    depth=2,
    heads=4,
    dim_head=64,
    mlp_dim=512,
    lambda_I=9.0,
    lambda_V=0.5,
    lambda_C=0.5,
    use_uncertainty_weighting=True,
    use_gradnorm=False,
    shrinkage_alpha=None,
    epsilon=1e-4
)

# Train with mixed precision
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    for x, _ in train_loader:
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            acc, loss = model.set_forward_loss(x)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```

## Expected Performance

### Improvements Over Baseline

1. **Cosine Attention**: +5-20 points vs. scaled dot-product attention
2. **VIC Regularization**: +2-8 points (largest on 5-shot, long-tailed data)
3. **Mahalanobis Distance**: +1-3 points vs. Euclidean distance
4. **Dynamic Weighting**: Improved stability and consistency

### Dataset-Specific Expectations

| Dataset | Baseline Acc | With EnhancedFSCT | Improvement |
|---------|-------------|-------------------|-------------|
| mini-ImageNet 1-shot | ~50% | ~55-60% | +5-10% |
| mini-ImageNet 5-shot | ~70% | ~73-78% | +3-8% |
| CIFAR-FS 1-shot | ~60% | ~67-72% | +7-12% |
| CIFAR-FS 5-shot | ~80% | ~83-87% | +3-7% |
| CUB 1-shot | ~75% | ~81-85% | +6-10% |
| CUB 5-shot | ~90% | ~92-94% | +2-4% |

Note: Gains are largest on challenging datasets and with limited shots.

## Ablation Guidelines

To understand component contributions:

1. **Start with baseline**: FSCT_cosine (cosine attention only)
2. **Add learnable prototypes**: Observe weight adaptation
3. **Add Mahalanobis distance**: Check covariance benefits
4. **Add VIC losses individually**: V → C → full VIC
5. **Test dynamic weighting**: Compare uncertainty vs. fixed vs. GradNorm

## Troubleshooting

### NaN/Inf in Losses
- Increase `epsilon` (try 1e-3)
- Check shrinkage_alpha (try fixed 0.2)
- Enable gradient clipping
- Reduce learning rate

### Out of Memory
- Enable `use_amp=True`
- Reduce `n_query` (e.g., 4 or 6)
- Use smaller backbone
- Reduce `mlp_dim` (e.g., 256)

### Poor Convergence
- Check VIC weight ratios (should be ~9:0.5:0.5)
- Increase `lambda_I` if accuracy is low
- Increase `lambda_V` or `lambda_C` if features collapse
- Try different dynamic weighting strategy

### Slow Training
- Enable mixed precision (`use_amp=True`)
- Reduce `depth` to 1 block
- Use smaller image size (84×84)
- Reduce attention heads (2 instead of 4)

## References

1. **FS-CT**: Nguyen et al., "Enhancing Few-Shot Image Classification With Cosine Transformer", IEEE Access 2023
2. **ProFONet**: ProFONet paper on VIC regularization for few-shot learning
3. **VICReg**: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization", ICLR 2022
4. **GradNorm**: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing", ICML 2018
5. **Mahalanobis-FSL**: Papers on Mahalanobis distance for few-shot classification
6. **ProtoNets**: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017

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
