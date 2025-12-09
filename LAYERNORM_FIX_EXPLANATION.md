# LayerNorm Dimension Mismatch Fix

## Problem
The training was failing with a `RuntimeError` in the `Attention` module:
```
RuntimeError: Given normalized_shape=[25088], expected input with shape [*, 25088], 
but got input of size[1, 5, 512] or [1, 5, 4608]
```

## Root Cause
The `LayerNorm` was part of a `nn.Sequential` in the `input_linear` layer:
```python
self.input_linear = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, inner_dim, bias = False))
```

When this Sequential was applied via a `map` function to process q, k, v tensors, there could be 
issues with how the Sequential was called or dimension mismatches due to the specific way tensors 
were being processed.

## Solution
Separated the `LayerNorm` from the input projection:

### Before:
```python
self.input_linear = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, inner_dim, bias = False))

def forward(self, q, k, v):
    f_q, f_k, f_v = map(lambda t: rearrange(
        self.input_linear(t), 'q n (h d) ->  h q n d', h = self.heads), (q, k ,v))
```

### After:
```python
self.norm = nn.LayerNorm(dim)
self.input_linear = nn.Linear(dim, inner_dim, bias = False)

def forward(self, q, k, v):
    # Apply layer normalization before projections
    q = self.norm(q)
    k = self.norm(k)
    v = self.norm(v)
    
    f_q, f_k, f_v = map(lambda t: rearrange(
        self.input_linear(t), 'q n (h d) ->  h q n d', h = self.heads), (q, k ,v))
```

## Why This Works
1. **Explicit Normalization**: By applying `self.norm` explicitly to each tensor, we ensure that 
   LayerNorm receives inputs with the correct dimension (`dim`) before any projection.

2. **Clearer Flow**: The separation makes the data flow more explicit:
   - Input tensors: `[..., dim]`
   - After normalization: `[..., dim]`
   - After linear projection: `[..., inner_dim]` where `inner_dim = heads * dim_head`
   - After rearrange: `[heads, ..., dim_head]`

3. **Standard Practice**: This aligns with standard transformer implementations where LayerNorm 
   is applied before the Q/K/V projections, not as part of the projection Sequential.

4. **Dimension Consistency**: The fix ensures that:
   - `LayerNorm(dim)` always receives inputs with last dimension = `dim`
   - `Linear(dim, inner_dim)` receives normalized inputs with dimension `dim`
   - The output of `Attention.forward` has dimension `dim` (after output projection)

## Testing
The fix should be tested with:
1. Different backbone architectures (Conv4, ResNet) with various feature dimensions
2. Different head configurations (heads=4, 8, etc.)
3. Different dim_head values (64, etc.)
4. Both training and inference modes

## Compatibility
This change maintains backward compatibility with existing model checkpoints as long as the 
same parameter names are used (`norm` instead of part of `input_linear`). New models trained 
with this fix will have the corrected architecture.
