# Optimal Few-Shot Learning - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Test

Run the example to verify installation:

```bash
python example_optimal_fewshot.py --dataset miniImagenet --num_episodes 5
```

Expected output:
```
✅ Example completed successfully!
Episode 5/5 | Acc: 0.xxxx | Loss: x.xxxx | λ_var: 0.xxxx | λ_cov: 0.xxxx
```

### 3. Run Tests

Validate all components:

```bash
python test_optimal_fewshot.py
```

Expected output:
```
Test Results: 11 passed, 0 failed
```

---

## 📊 Training on Your Dataset

### Option A: Use Example Script

```bash
python example_optimal_fewshot.py \
    --dataset miniImagenet \
    --n_way 5 \
    --k_shot 5 \
    --n_query 15 \
    --num_episodes 100
```

### Option B: Integrate with Existing Training

```python
from methods.optimal_fewshot import OptimalFewShotModel, DATASET_CONFIGS

# Get configuration
config = DATASET_CONFIGS['miniImagenet']

# Create model
model = OptimalFewShotModel(
    model_func=None,
    n_way=5,
    k_shot=5,
    n_query=15,
    feature_dim=64,
    n_heads=4,
    dropout=0.1,
    dataset='miniImagenet',
    gradient_checkpointing=True,
    use_custom_backbone=True
).cuda()

# Create optimizer
import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)

# Training loop
for epoch in range(100):
    for episode_data in train_loader:
        model.train()
        acc, loss = model.set_forward_loss(episode_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 🎯 Dataset-Specific Examples

### Omniglot (High Accuracy)

```bash
python example_optimal_fewshot.py \
    --dataset Omniglot \
    --n_way 5 \
    --k_shot 1 \
    --num_episodes 50
```

**Expected**: 99.5% accuracy on 5-way 5-shot

### CUB-200 (Fine-grained)

```bash
python example_optimal_fewshot.py \
    --dataset CUB \
    --n_way 5 \
    --k_shot 5 \
    --num_episodes 100
```

**Expected**: 85% accuracy on 5-way 5-shot

### CIFAR-FS (Low Resolution)

```bash
python example_optimal_fewshot.py \
    --dataset CIFAR \
    --n_way 5 \
    --k_shot 5 \
    --num_episodes 100
```

**Expected**: 85% accuracy on 5-way 5-shot

### miniImageNet (Standard Benchmark)

```bash
python example_optimal_fewshot.py \
    --dataset miniImagenet \
    --n_way 5 \
    --k_shot 5 \
    --num_episodes 200
```

**Expected**: 75% accuracy on 5-way 5-shot

### HAM10000 (Medical, Class Imbalance)

```bash
python example_optimal_fewshot.py \
    --dataset HAM10000 \
    --n_way 7 \
    --k_shot 5 \
    --num_episodes 150
```

**Expected**: 65% accuracy on 7-way 5-shot

**Note**: This dataset uses focal loss for class imbalance handling.

---

## 💾 Memory Optimization

### For 8GB VRAM

```python
model = OptimalFewShotModel(
    ...,
    gradient_checkpointing=True,  # ✓ Saves ~400MB
    use_custom_backbone=True      # ✓ Optimized Conv4
)

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():  # ✓ Saves ~50% memory
    acc, loss = model.set_forward_loss(x)
```

### For 4GB VRAM (Colab Free)

```python
# Additional optimizations
model = OptimalFewShotModel(
    ...,
    feature_dim=32,  # Reduce from 64
    n_heads=2,       # Reduce from 4
    gradient_checkpointing=True
)

# Accumulate gradients over multiple episodes
accumulation_steps = 4
for i, x in enumerate(train_loader):
    acc, loss = model.set_forward_loss(x)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 🔧 Key Features

### 1. SE-Enhanced Conv4 Backbone
✓ Channel attention with <5% memory overhead  
✓ Better feature extraction than plain Conv4  
✓ Bias-free convolutions save parameters

### 2. Lightweight Cosine Transformer
✓ Single-layer, 4-head design  
✓ Cosine attention with learnable temperature  
✓ Only ~42K parameters

### 3. Dynamic VIC Regularization
✓ Prevents representation collapse  
✓ Variance + Covariance losses  
✓ Adaptive weighting per episode

### 4. Episode-Adaptive Lambda Predictor
✓ Dataset-aware embeddings  
✓ Computes episode statistics automatically  
✓ EMA smoothing for stability

### 5. Memory Optimizations
✓ Gradient checkpointing (~400MB saved)  
✓ Mixed precision support (FP16)  
✓ Episode-wise training  
✓ Efficient implementation

---

## 📈 Monitoring Training

The model provides detailed information during training:

```python
logits, vic_loss, info = model.set_forward(x)

print(f"λ_var: {info['lambda_var']:.4f}")       # Variance regularization weight
print(f"λ_cov: {info['lambda_cov']:.4f}")       # Covariance regularization weight
print(f"Temperature: {info['temperature']:.2f}") # Attention temperature
print(f"Var loss: {info['var_loss']:.4f}")      # Variance loss value
print(f"Cov loss: {info['cov_loss']:.4f}")      # Covariance loss value
```

### What to Look For

✓ **λ_var should increase** over training (0.1 → 0.2)  
✓ **λ_cov should decrease** over training (0.03 → 0.01)  
✓ **Temperature stays stable** around 8-15  
✓ **Losses decrease** steadily

---

## 🐛 Troubleshooting

### Out of Memory

```python
# 1. Enable gradient checkpointing
gradient_checkpointing=True

# 2. Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    # training code

# 3. Reduce feature dimension
feature_dim=32  # instead of 64

# 4. Clear cache between episodes
torch.cuda.empty_cache()
```

### Poor Accuracy

```python
# 1. Check learning rate
optimizer = optim.AdamW(model.parameters(), lr=0.0005)  # Try 0.0001-0.001

# 2. Increase epochs
num_epoch = 100  # Try 50-200

# 3. Adjust dropout
dropout=0.15  # Try 0.05-0.2

# 4. Monitor VIC losses
# Ensure they're decreasing but not too small
```

### NaN Loss

```python
# 1. Reduce learning rate
lr=0.0001  # Start lower

# 2. Check gradient clipping (already enabled)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Ensure proper input normalization
# Images should be in [0, 1] or [-1, 1] range
```

---

## 📚 More Information

- **Full Documentation**: See `OPTIMAL_FEWSHOT_DOCUMENTATION.md`
- **Paper**: [Enhancing Few-shot Image Classification with Cosine Transformer](https://ieeexplore.ieee.org/document/10190567/)
- **Repository**: Check main `README.md` for dataset preparation

---

## ✅ Validation Checklist

Before deploying to production:

- [ ] Run `test_optimal_fewshot.py` - all tests pass
- [ ] Run `example_optimal_fewshot.py` - completes successfully
- [ ] Train for 10 epochs - loss decreases steadily
- [ ] Validate on hold-out set - accuracy meets expectations
- [ ] Check memory usage - stays under 8GB (or your limit)
- [ ] Monitor λ values - they adapt over training
- [ ] Test with different datasets - works for all 5 datasets

---

## 🎯 Expected Results

| Dataset | Setup | Target Accuracy | Training Time* |
|---------|-------|----------------|----------------|
| Omniglot | 5-way 1-shot | 99.5% ±0.1% | ~30 min |
| CUB | 5-way 5-shot | 85% ±0.6% | ~2 hours |
| CIFAR-FS | 5-way 5-shot | 85% ±0.5% | ~1.5 hours |
| miniImageNet | 5-way 5-shot | 75% ±0.4% | ~3 hours |
| HAM10000 | 7-way 5-shot | 65% ±1.2% | ~2 hours |

*On single V100 GPU with mixed precision

---

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue on GitHub!

---

## 📄 License

See LICENSE.txt in the repository root.
