# Formula Implementation Guide

This document provides a comprehensive mathematical formulation of the Few-Shot Cosine Transformer and Optimal Few-Shot Learning algorithms implemented in this repository.

## Table of Contents
- [Core Components](#core-components)
- [Cosine Attention Mechanism](#cosine-attention-mechanism)
- [Few-Shot Cosine Transformer](#few-shot-cosine-transformer)
- [Optimal Few-Shot Learning](#optimal-few-shot-learning)
- [Loss Functions](#loss-functions)

---

## Core Components

### 1. Feature Extraction

The backbone feature extractor (Conv4 or ResNet) transforms input images into feature representations:

```
φ: X → Z
```

Where:
- **X ∈ ℝ^(C×H×W)**: Input image with C channels, height H, width W
- **Z ∈ ℝ^d**: Feature embedding of dimension d

For Conv4 with 4 layers:
```
z = φ(x) = f₄ ∘ f₃ ∘ f₂ ∘ f₁(x)

where fᵢ(x) = MaxPool(ReLU(BatchNorm(Conv(x))))
```

### 2. L2 Normalization

Feature normalization for stable training:

```
ẑ = z / ||z||₂ = z / √(Σᵢ zᵢ²)
```

### 3. Prototypical Representation

For N-way K-shot learning, the prototype for class c is the mean of support features:

```
pᶜ = (1/K) Σₖ₌₁ᴷ zₛᶜ'ᵏ

where zₛᶜ'ᵏ is the k-th support feature of class c
```

---

## Cosine Attention Mechanism

### Standard Dot-Product Attention (Baseline)

The traditional scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

Where:
- **Q ∈ ℝ^(n×dₖ)**: Query matrix
- **K ∈ ℝ^(m×dₖ)**: Key matrix  
- **V ∈ ℝ^(m×dᵥ)**: Value matrix
- **dₖ**: Dimension of keys
- **√dₖ**: Scaling factor

### Cosine Attention (Proposed)

Our proposed cosine-based attention mechanism:

```
Attention_cos(Q, K, V) = softmax(cos_sim(Q, K) / τ)V
```

Where cosine similarity is defined as:

```
cos_sim(Q, K) = (Q̂Kᵀ)

with Q̂ = Q / ||Q||₂ and K̂ = K / ||K||₂
```

The attention weights are:

```
α = softmax((Q̂K̂ᵀ) / τ)

where τ is a learnable temperature parameter
```

**Key Properties:**
- Range: cos_sim ∈ [-1, 1] (bounded)
- Invariant to feature magnitude
- Better correlation stability
- Learnable temperature τ controls sharpness

### Multi-Head Cosine Attention

Extend to multiple heads for richer representations:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)Wᴼ

where headᵢ = Attention_cos(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)
```

Parameters:
- **h**: Number of attention heads (typically 4 or 8)
- **Wᵢᵠ, Wᵢᴷ, Wᵢⱽ**: Projection matrices for head i
- **Wᴼ**: Output projection matrix

---

## Few-Shot Cosine Transformer

### Complete Architecture

The Few-Shot Cosine Transformer processes support and query sets:

```
ŷ = FSCT(Xₛ, Xᵩ)
```

#### Step 1: Feature Extraction
```
Zₛ = {φ(xₛⁱ'ʲ) | i ∈ [1,N], j ∈ [1,K]}  (Support features)
Zᵩ = {φ(xᵩⁱ) | i ∈ [1,Q]}                (Query features)
```

#### Step 2: Learnable Prototypical Embedding
```
P = PrototypeNet(Zₛ)

where P = {pᶜ | c ∈ [1,N]}

pᶜ = LayerNorm(MLP(mean(Zₛᶜ)))
```

#### Step 3: Cosine Transformer Block

The transformer processes both support and query features:

```
Z' = TransformerBlock(Z)
```

Where the transformer block consists of:

**a) Multi-Head Cosine Attention:**
```
Z_attn = Z + MultiHeadCosineAttn(LN(Z))
```

**b) Feed-Forward Network:**
```
Z' = Z_attn + FFN(LN(Z_attn))

FFN(x) = W₂·ReLU(W₁·x + b₁) + b₂
```

#### Step 4: Cosine Classification

Final prediction using cosine similarity:

```
s(xᵩ, c) = cos_sim(zᵩ, pᶜ) / τ

logits = [s(xᵩ, 1), s(xᵩ, 2), ..., s(xᵩ, N)]

ŷ = argmax(softmax(logits))
```

---

## Optimal Few-Shot Learning

### 1. SE (Squeeze-and-Excitation) Block

Channel attention mechanism for adaptive feature recalibration:

```
SE(X) = X ⊙ σ(W₂·ReLU(W₁·GAP(X)))
```

Where:
- **GAP(X)**: Global Average Pooling: `(1/(H×W)) Σᵢⱼ Xᵢⱼ`
- **W₁ ∈ ℝ^(C/r×C)**: Squeeze layer (reduction ratio r=4)
- **W₂ ∈ ℝ^(C×C/r)**: Excitation layer
- **σ**: Sigmoid activation
- **⊙**: Element-wise multiplication

Channel weights:
```
wᶜ = σ(W₂·ReLU(W₁·zᶜ_global))

where zᶜ_global = (1/(H×W)) Σᵢⱼ Xᶜᵢⱼ
```

### 2. Optimized Conv4 Backbone

Enhanced Conv4 with SE blocks:

```
Conv4_SE(x) = SE(f₄(SE(f₃(SE(f₂(SE(f₁(x))))))))
```

Each layer fᵢ:
```
fᵢ(x) = Dropout(MaxPool(ReLU(BatchNorm(Conv(x)))))
```

### 3. Dynamic VIC Regularization

**VIC Loss = Variance Loss + Covariance Loss**

#### Variance Loss (Inter-Class Separation)
Maximizes distance between class prototypes:

```
L_var = (1/N(N-1)) Σᵢ≠ⱼ cos_sim(pᵢ, pⱼ)
```

Encourages prototypes to be dissimilar (orthogonal in feature space).

#### Covariance Loss (Dimension Decorrelation)
Prevents feature dimension redundancy:

```
L_cov = (1/D²) Σᵢ≠ⱼ (Cov(P)ᵢⱼ)²

where Cov(P) = (P - μ)ᵀ(P - μ) / (N-1)
```

And μ is the mean prototype: `μ = (1/N) Σᵢ pᵢ`

#### Total VIC Loss
```
L_VIC = λ_var · L_var + λ_cov · L_cov
```

### 4. Episode-Adaptive Lambda Predictor

Dynamically computes λ_var and λ_cov based on episode statistics:

**Episode Statistics:**
```
1. Intra-class variance:
   σ²_intra = (1/N) Σᶜ Var(Zₛᶜ)

2. Inter-class separation:
   sep_inter = 1 - (1/(N(N-1))) Σᵢ≠ⱼ cos_sim(pᵢ, pⱼ)

3. Domain shift:
   shift = 1 - cos_sim(mean(Zₛ), mean(Zᵩ))

4. Support diversity:
   div_s = (1/N) Σᶜ std(Zₛᶜ)

5. Query diversity:
   div_q = std(Zᵩ)
```

**Lambda Prediction:**
```
stats = [σ²_intra, sep_inter, shift, div_s, div_q]
embedding = DatasetEmbed(dataset_id)
x = Concat(stats, embedding)

[λ_var, λ_cov] = Sigmoid(MLP(x)) · 0.5
```

**EMA Smoothing:**
```
λ̄_t = β·λ̄_{t-1} + (1-β)·λ_t

where β = 0.9 (momentum)
```

Final clamped values:
```
λ_var ∈ [0.05, 0.3]
λ_cov ∈ [0.005, 0.1]
```

### 5. Complete Optimal Few-Shot Pipeline

**Forward Pass:**
```
Input: Xₛ (support), Xᵩ (query)

1. Feature extraction:
   Zₛ = Conv4_SE(Xₛ)
   Zᵩ = Conv4_SE(Xᵩ)

2. Projection to transformer space:
   Z'ₛ = Proj(Zₛ)
   Z'ᵩ = Proj(Zᵩ)

3. Transformer encoding:
   Z''ₛ, Z''ᵩ = CosineTransformer([Z'ₛ, Z'ᵩ])

4. Prototype computation:
   P = {mean(Z''ₛᶜ) | c ∈ [1,N]}
   P̂ = Normalize(P)

5. Adaptive lambda:
   λ_var, λ_cov = EpisodeLambda(P, Z''ₛ, Z''ᵩ)

6. Classification:
   logits = (Normalize(Z''ᵩ) · P̂ᵀ) · τ
```

---

## Loss Functions

### 1. Cross-Entropy Loss with Label Smoothing

Standard cross-entropy:
```
L_CE = -Σᵢ yᵢ log(ŷᵢ)
```

With label smoothing (ε = 0.1):
```
y'ᵢ = (1-ε)·yᵢ + ε/N

L_CE_smooth = -Σᵢ y'ᵢ log(ŷᵢ)
```

### 2. Focal Loss (for Imbalanced Data)

Addresses class imbalance by down-weighting easy examples:

```
L_focal = -α·(1-pₜ)^γ·log(pₜ)

where pₜ = {
    p      if y = 1
    1-p    otherwise
}
```

Parameters:
- **α = 0.25**: Balancing factor
- **γ = 2.0**: Focusing parameter

### 3. Total Training Loss

Complete loss for Optimal Few-Shot:

```
L_total = L_CE + L_VIC

       = L_CE + λ_var·L_var + λ_cov·L_cov
```

For imbalanced datasets:
```
L_total = L_focal + λ_var·L_var + λ_cov·L_cov
```

---

## Gradient Checkpointing

Memory-efficient training using gradient checkpointing:

```
During forward:
  Store only selected activations
  
During backward:
  Recompute intermediate activations
```

Memory savings:
```
Memory_saved ≈ (L-1)/L × Memory_activations

where L is number of checkpointed layers
```

---

## Computational Complexity

### Standard Attention
```
O(n² · d + n · d²)
```

### Cosine Attention  
```
O(n² · d + n · d)  (slightly faster due to normalization)
```

### Overall Model Complexity
```
O(B · (C_conv + C_transformer + C_classification))

where:
- B: Batch size (episodes)
- C_conv: O(K² · C · H · W · D) for convolutions
- C_transformer: O(N² · d²) for attention
- C_classification: O(N · d) for cosine classification
```

---

## Hyperparameter Guidelines

### Cosine Transformer
- **Temperature τ**: 0.05 - 0.2 (learnable, initialized at 0.05)
- **Number of heads h**: 4 or 8
- **Transformer dimension d**: 64 or 128
- **Dropout**: 0.1 - 0.2

### VIC Regularization
- **λ_var**: 0.05 - 0.3 (adaptive, typical: 0.1)
- **λ_cov**: 0.005 - 0.1 (adaptive, typical: 0.01)
- **EMA momentum β**: 0.9

### Optimization
- **Learning rate**: 1e-4 to 1e-3
- **Batch size**: 4-16 episodes
- **Optimizer**: Adam with (β₁=0.9, β₂=0.999)
- **Weight decay**: 1e-4

---

## Key Innovations Summary

1. **Cosine Attention**: Replaces dot-product with cosine similarity for bounded, magnitude-invariant attention
2. **SE Blocks**: Channel-wise attention with <5% overhead
3. **VIC Regularization**: Ensures well-separated, decorrelated prototypes
4. **Adaptive Lambda**: Episode-aware hyperparameter tuning
5. **Gradient Checkpointing**: Reduces memory by ~400MB
6. **Label Smoothing**: Prevents overconfidence

---

## References

### Mathematical Foundations
- **Attention Mechanism**: Vaswani et al., "Attention Is All You Need" (2017)
- **Cosine Similarity**: Introduced in information retrieval and adapted for neural attention
- **Prototypical Networks**: Snell et al., "Prototypical Networks for Few-shot Learning" (2017)
- **SE Blocks**: Hu et al., "Squeeze-and-Excitation Networks" (2018)
- **VIC Regularization**: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization" (2022)

### Implementation
- **Few-Shot Cosine Transformer**: Nguyen et al., "Enhancing Few-Shot Image Classification With Cosine Transformer" (2023)
- **Cross Transformers**: Doersch et al., "CrossTransformers: spatially-aware few-shot transfer" (2020)

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| N | Number of ways (classes) |
| K | Number of shots (support examples per class) |
| Q | Number of query examples per class |
| d | Feature dimension |
| h | Number of attention heads |
| τ | Temperature parameter |
| P | Set of prototypes |
| Zₛ | Support features |
| Zᵩ | Query features |
| λ_var | Variance loss weight |
| λ_cov | Covariance loss weight |
| ⊙ | Element-wise multiplication |
| ∘ | Function composition |
| ‖·‖₂ | L2 norm |

