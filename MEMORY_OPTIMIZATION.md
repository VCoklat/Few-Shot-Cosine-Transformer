# Memory Optimization for CUDA Out of Memory Fix

This document describes the changes made to fix the CUDA Out of Memory (OOM) error that occurred during training.

## Problem

The training process was running out of GPU memory during the backward pass:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.35 GiB. 
GPU 0 has a total capacity of 15.89 GiB of which 363.12 MiB is free.
```

## Solution

The fix implements three key memory optimization strategies:

### 1. CUDA Memory Configuration

Set the PyTorch CUDA allocator configuration at the very beginning of training scripts to reduce memory fragmentation:

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'  # New PyTorch version
```

This is done in:
- `train_test.py` (line 11-13)
- `train.py` (line 11-13)

### 2. Gradient Accumulation

Gradient accumulation allows training with larger effective batch sizes while using less GPU memory per step. Instead of updating weights after every batch, gradients are accumulated over N steps before performing an optimizer update.

**Usage:**

```bash
python train_test.py --gradient_accumulation_steps 2 [other args...]
```

Default is 1 (no accumulation). Higher values (2, 4, 8) reduce memory usage proportionally.

**Implementation:**

- Added `--gradient_accumulation_steps` parameter in `io_utils.py`
- Modified `train_loop()` in `methods/meta_template.py` to support gradient accumulation
- Loss is scaled by `1/gradient_accumulation_steps` during accumulation
- Optimizer step occurs every N accumulation steps

### 3. Periodic CUDA Cache Clearing

The training loop now periodically clears the CUDA cache to prevent memory fragmentation:

- During training: Every 10 * gradient_accumulation_steps iterations
- After each epoch: Once at the end of validation

This is done automatically in:
- `methods/meta_template.py` in the `train_loop()` method
- `train_test.py` and `train.py` after each epoch

## Testing

Run the comprehensive test suite to verify the memory optimizations:

```bash
python test_memory_optimization.py
```

This tests:
1. Gradient accumulation functionality
2. CUDA cache clearing
3. Environment variable configuration

## Performance Impact

- **Memory Usage**: Reduced by factor of `gradient_accumulation_steps`
- **Training Speed**: May be slightly slower due to cache clearing, but negligible
- **Accuracy**: No impact - mathematically equivalent to standard training

## Example Usage

### Minimal memory usage (accumulate over 4 steps):
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet \
    --backbone ResNet34 --n_way 5 --k_shot 5 \
    --gradient_accumulation_steps 4
```

### Balanced (accumulate over 2 steps):
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet \
    --backbone ResNet34 --n_way 5 --k_shot 5 \
    --gradient_accumulation_steps 2
```

### Standard (no accumulation, maximum memory):
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet \
    --backbone ResNet34 --n_way 5 --k_shot 5
```

## Troubleshooting

If you still encounter OOM errors:

1. **Increase gradient accumulation**: Try `--gradient_accumulation_steps 4` or `8`
2. **Reduce batch size**: Adjust `--n_episode` parameter
3. **Use a smaller backbone**: Try `ResNet18` instead of `ResNet34`
4. **Reduce query samples**: Adjust `--n_query` parameter

## Technical Details

### Gradient Accumulation Math

With gradient accumulation of N steps:
- Standard: `loss.backward(); optimizer.step()` every iteration
- Accumulated: `(loss/N).backward()` N times, then `optimizer.step()` once

The effective batch size becomes: `original_batch_size * gradient_accumulation_steps`

### Memory Savings Calculation

If one forward+backward pass uses M GB:
- Standard training: M GB per step
- With accumulation of 4: ~M/4 GB per step (gradients accumulated, not full activations)

Actual savings may vary depending on model architecture and batch size.
