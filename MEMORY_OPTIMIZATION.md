# Memory Optimization Guide

This document describes the memory optimizations implemented to address CUDA out-of-memory (OOM) errors during training.

## Problem

During training with large models (especially ProFOCT with VIC regularization) and high-resolution images, the following CUDA OOM error was encountered:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.35 GiB. 
GPU 0 has a total capacity of 15.89 GiB of which 375.12 MiB is free.
```

## Solutions Implemented

### 1. Gradient Accumulation

**What it does:** Splits each training step into multiple smaller forward/backward passes, accumulating gradients before updating model weights.

**Benefits:**
- Reduces peak memory usage during backward pass
- Allows training with effectively larger batch sizes on limited memory
- No loss in model quality when properly configured

**Usage:**
```bash
python train_test.py --gradient_accumulation_steps 2 [other args]
```

**Default:** 2 steps (memory-efficient)
**Recommended values:**
- For 16GB GPU: 2-4 steps
- For 8GB GPU: 4-8 steps
- For larger GPUs (24GB+): 1 step (no accumulation needed)

### 2. Automatic Mixed Precision (AMP)

**What it does:** Uses FP16 (half precision) for most operations while keeping FP32 for critical operations to maintain numerical stability.

**Benefits:**
- ~50% reduction in memory usage
- ~2-3x faster training on modern GPUs (with Tensor Cores)
- Minimal impact on model accuracy with gradient scaling

**Usage:**
```bash
python train_test.py --use_amp 1 [other args]
```

**Default:** Enabled (1)
**Note:** Requires CUDA-capable GPU. Automatically disabled on CPU.

### 3. Aggressive Memory Clearing

**Improvements made:**
- More frequent CUDA cache clearing (every 10-50 iterations)
- Explicit deletion of intermediate tensors in ProFOCT
- Cache clearing after each training epoch
- Detaching tensors to prevent gradient graph retention

**Location:** Implemented in:
- `methods/meta_template.py`: `train_loop()` method
- `methods/ProFOCT.py`: `set_forward_loss()` method
- `train_test.py`: `train()` function

### 4. Memory-Efficient ProFOCT

**Optimizations for ProFOCT's VIC regularization:**
- Detach intermediate tensors in VIC weight updates to prevent graph retention
- Detach attention outputs when computing VIC on attention (optional feature)
- Explicit tensor deletion after loss computation
- Compute accuracy in no_grad context

**Usage:**
ProFOCT-specific optimizations are automatic. To reduce memory further, disable VIC on attention:
```bash
python train_test.py --method ProFOCT_cosine --use_vic_on_attention 0 [other args]
```

## Configuration Examples

### Minimal Memory Usage (for small GPUs ~8GB)
```bash
python train_test.py \
  --method FSCT_cosine \
  --dataset miniImagenet \
  --backbone ResNet18 \
  --gradient_accumulation_steps 4 \
  --use_amp 1 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 8
```

### Balanced (for medium GPUs ~16GB)
```bash
python train_test.py \
  --method ProFOCT_cosine \
  --dataset miniImagenet \
  --backbone ResNet34 \
  --gradient_accumulation_steps 2 \
  --use_amp 1 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 16
```

### Maximum Performance (for large GPUs ~24GB+)
```bash
python train_test.py \
  --method ProFOCT_cosine \
  --dataset miniImagenet \
  --backbone ResNet34 \
  --FETI 1 \
  --gradient_accumulation_steps 1 \
  --use_amp 1 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 16
```

## Environment Variable

The following environment variable is automatically set in `train_test.py`:
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

This helps PyTorch manage memory fragmentation more efficiently.

## Testing Memory Optimizations

Run the test script to validate that memory optimizations work correctly:

```bash
python test_memory_optimizations.py
```

This tests:
- Gradient accumulation functionality
- AMP forward/backward passes
- ProFOCT memory optimizations
- CUDA cache clearing

## Troubleshooting

### Still getting OOM errors?

1. **Increase gradient accumulation steps:**
   ```bash
   --gradient_accumulation_steps 4  # or 8
   ```

2. **Reduce query samples:**
   ```bash
   --n_query 8  # instead of 16
   ```

3. **Use smaller backbone:**
   ```bash
   --backbone ResNet18  # instead of ResNet34
   ```

4. **Disable VIC on attention (ProFOCT only):**
   ```bash
   --use_vic_on_attention 0
   ```

5. **Monitor GPU memory:**
   ```bash
   watch -n 1 nvidia-smi
   ```

### AMP causing numerical instability?

Disable AMP if you notice training instability:
```bash
--use_amp 0
```

### Gradient accumulation slowing down training?

While gradient accumulation reduces memory, it may slow down training slightly. If you have enough memory:
```bash
--gradient_accumulation_steps 1  # Disable accumulation
```

## Performance Impact

Expected changes with default settings (gradient_accumulation_steps=2, use_amp=1):

- **Memory usage:** ~40-60% reduction
- **Training speed:** Similar or 10-30% faster (due to AMP)
- **Model accuracy:** No significant change (<0.5% difference)

## References

- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Gradient Accumulation Tutorial](https://kozodoi.me/blog/20210219/gradient-accumulation)
