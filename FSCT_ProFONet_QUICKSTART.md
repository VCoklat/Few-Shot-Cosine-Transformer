# Quick Start: FSCT_ProFONet Method

## What is FSCT_ProFONet?

A hybrid few-shot classification algorithm combining:
- **FS-CT**: Cosine Attention Transformer
- **ProFONet**: VIC Regularization (Variance-Invariance-Covariance)
- **Dynamic Weight Scheduling**: Adaptive regularization during training

**Goal**: >20% accuracy improvement over baseline, optimized for 8GB VRAM.

## Quick Usage

### 1. Train with FSCT_ProFONet

```bash
# 5-way 5-shot on miniImagenet
python train.py \
  --method FSCT_ProFONet \
  --dataset miniImagenet \
  --backbone Conv4 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 10 \
  --num_epoch 50
```

### 2. Test the Model

```bash
python test.py \
  --method FSCT_ProFONet \
  --dataset miniImagenet \
  --backbone Conv4 \
  --n_way 5 \
  --k_shot 5
```

## Key Features

✅ **VIC Regularization**: Prevents representation collapse  
✅ **Dynamic Weights**: λ_V increases, λ_C decreases during training  
✅ **Cosine Attention**: Stable attention without softmax  
✅ **Memory Optimized**: Gradient checkpointing + mixed precision  
✅ **Gradient Clipping**: Training stability (max_norm=1.0)  

## What's Different?

### Compared to FSCT_cosine:
- ➕ VIC Regularization (3 complementary losses)
- ➕ Dynamic weight adjustment
- ➕ Gradient clipping
- ➕ Memory optimizations

### Compared to ProFONet:
- ➕ Cosine attention (not dot-product)
- ➕ No softmax in attention
- ➕ Learnable prototype weights
- ➕ Transformer architecture

## Configuration

### Default (Optimized for 8GB VRAM):
- Attention heads: 4
- Head dimension: 160
- Query samples: 10
- Gradient checkpointing: Enabled (CUDA)
- Mixed precision: Enabled (CUDA)

### VIC Weights (Dynamic):
- λ_V: 0.5 → 0.65 (increases)
- λ_I: 9.0 (constant)
- λ_C: 0.5 → 0.40 (decreases)

## Testing

```bash
# Run unit tests
python test_fsct_profonet.py

# Run integration tests
python test_integration.py
```

## Documentation

See `FSCT_ProFONet_DOCUMENTATION.md` for detailed documentation.

## Results

Expected improvements:
- Better generalization through VIC regularization
- More stable training with dynamic weights
- Improved few-shot accuracy (target: >20% improvement)

## Troubleshooting

**Out of Memory?**
- Reduce `--n_query` to 8
- Use Conv4 instead of ResNet
- Ensure gradient checkpointing is enabled

**Training Unstable?**
- Check gradient clipping is working
- Monitor loss components (V, I, C)
- Verify dynamic weights are updating

## Citation

```bibtex
@article{nguyen2023FSCT,
  author={Nguyen, Quang-Huy and Nguyen, Cuong Q. and Le, Dung D. and Pham, Hieu H.},
  journal={IEEE Access}, 
  title={Enhancing Few-Shot Image Classification With Cosine Transformer}, 
  year={2023}
}
```
