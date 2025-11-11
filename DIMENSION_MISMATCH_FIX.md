# Sequence Dimension Mismatch Fix

## Problem
When training with the FSCT_cosine method on the CUB dataset (5-way, 5-shot, 16 queries), a warning was displayed repeatedly:
```
Warning: Sequence dimension mismatch - dots expects seq_k=80 but f_v has seq_v=1
```

This occurred because in few-shot learning scenarios:
- Support prototypes (q) have shape `[1, n_way, d]` (batch=1)
- Query samples (k, v) have shape `[n_way*n_query, 1, d]` (batch=80 for 5-way 16-query)

After transformation through the attention mechanism:
- `f_q` had shape `[heads, 1, n_way, head_dim]`
- `f_k` and `f_v` had shape `[heads, 80, 1, head_dim]`

This batch dimension mismatch caused incorrect broadcasting during attention computation, resulting in misaligned sequence dimensions between the attention weights (dots) and the values (f_v).

## Solution
Added batch dimension alignment logic in the `Attention.forward()` method (lines 685-706 in `methods/transformer.py`):

1. **Detection**: After the `rearrange` operation, check if `f_q` and `f_k` have different batch dimensions
2. **Reshaping**: When q has batch=1 and k/v have batch>1 (few-shot cross-attention case):
   - Reshape k and v from `[heads, batch_k, seq_k, dim]` to `[heads, 1, batch_k*seq_k, dim]`
   - This folds the batch dimension into the sequence dimension
   - Ensures k and v have the same batch dimension as q
   - The sequence dimension becomes `batch_k * seq_k`, which matches the attention computation

## Result
After the fix:
- `f_q = [heads, 1, 5, head_dim]` (batch=1, seq_q=5)
- `f_k = [heads, 1, 80, head_dim]` (batch=1, seq_k=80)
- `f_v = [heads, 1, 80, head_dim]` (batch=1, seq_v=80)
- Attention weights: `dots = [heads, 1, 5, 80]`
- Dimension check: `f_v.shape[2]=80 == dots.shape[3]=80` ✅
- No warning is displayed
- Correct attention computation: `dots @ f_v = [heads, 1, 5, 80] @ [heads, 1, 80, 64] = [heads, 1, 5, 64]` ✅

## Impact
- **Minimal code change**: Only added 22 lines of dimension handling code
- **No breaking changes**: The fix only activates when batch dimensions differ
- **Maintains backward compatibility**: Existing code paths unchanged
- **Fixes cross-attention**: Properly handles few-shot learning scenarios where support and query have different batch sizes
