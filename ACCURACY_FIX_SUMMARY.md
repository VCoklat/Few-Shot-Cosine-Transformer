# Accuracy Improvement Fix Summary

## Problem Statement

The model was achieving only ~20-21% validation accuracy on 5-way 1-shot classification, which is essentially random guessing (1/5 = 20%). The training showed:

- Training accuracy stuck at 20%
- Validation accuracy stuck at ~19-21%
- Extremely high losses (5000-60000+)
- Confusion matrix showing the model predicting almost everything as class 0

## Root Causes Identified

### 1. **Unsupported Method Name** (Moderate Impact)
- User was using `--method ProFOCT_cosine` which was not recognized by the code
- This should have caused an error, but the condition checking allowed it to pass through

### 2. **CRITICAL: Shape Mismatch in Score Computation** (HIGH IMPACT)
The most critical bug was in the `set_forward` method of `FewShotTransformer`:

**Before Fix:**
```python
def set_forward(self, x, is_feature=False):
    z_support, z_query = self.parse_feature(x, is_feature)
    z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
    z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
    z_query = z_query.contiguous().reshape(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)
    
    x, query = z_proto, z_query
    
    # Process through transformer layers
    for _ in range(self.depth):
        attn_output = self.ATTN(q=x, k=query, v=query, ...)
        x = attn_output + x
        x = self.FFN_forward(x) + x
    
    # BUG: Only returns scores for prototypes, not for each query!
    return self.final_linear_forward(x).squeeze()  # Shape: (n_way,) ❌
```

**The Issue:**
- After transformer processing, `x` had shape `(1, n_way, d)` (refined prototypes)
- The final output was squeezed to shape `(n_way,)` - only 5 values for a 5-way task
- But the target labels had shape `(n_way * n_query,)` = `(5 * 16,)` = 80 values
- CrossEntropyLoss expected shape `(80, 5)` but got `(5,)` or was broadcasted incorrectly

**Why This Caused 20% Accuracy:**
- All 80 queries were effectively getting the same 5 scores
- The model couldn't distinguish between different queries
- Predictions collapsed to always choosing class 0 (as seen in confusion matrix)
- This is exactly random guessing for 5-way classification

### 3. **Missing CLI Parameters** (Low Impact)
Several command-line parameters were missing from the argument parser:
- `gradient_accumulation_steps`
- `use_amp`
- `dynamic_vic`
- `vic_alpha`, `vic_beta`, `vic_gamma`
- `vic_attention_scale`
- `use_vic_on_attention`
- `distance_metric`

## Fixes Implemented

### Fix 1: Add ProFOCT Method Support
```python
# In io_utils.py
parser.add_argument('--method', default='FSCT_cosine', 
    help='CTX_softmax/CTX_cosine/FSCT_softmax/FSCT_cosine/ProFOCT_cosine/ProFOCT_softmax')

# In train_test.py
if params.method in ['FSCT_softmax', 'FSCT_cosine', 'ProFOCT_cosine', 'ProFOCT_softmax']:
    variant = 'cosine' if params.method in ['FSCT_cosine', 'ProFOCT_cosine'] else 'softmax'
    # ProFOCT is an alias for FSCT (Prototype Few-shot Cosine Transformer)
```

### Fix 2: **CRITICAL - Compute Proper Query-Prototype Scores**
```python
def set_forward(self, x, is_feature=False):
    z_support, z_query = self.parse_feature(x, is_feature)
    z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
    z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
    z_query = z_query.contiguous().reshape(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)

    x, query = z_proto, z_query

    # Process through transformer layers
    for _ in range(self.depth):
        attn_output = self.ATTN(q=x, k=query, v=query, ...)
        x = attn_output + x
        x = self.FFN_forward(x) + x

    # NEW: Compute scores for each query against each prototype
    proto_features = self.final_linear_forward(x)  # (1, n_way, dim_head)
    
    # Process each query through the same layers
    query_features = []
    for i in range(z_query.shape[0]):
        q_feat = self.final_linear_forward(z_query[i:i+1])  # (1, 1, dim_head)
        query_features.append(q_feat)
    query_features = torch.cat(query_features, dim=0)  # (n_way*n_query, 1, dim_head)
    
    # Compute scores: for each query, compute similarity with each prototype
    proto_features = proto_features.squeeze(0)  # (n_way, dim_head)
    query_features = query_features.squeeze(1)  # (n_way*n_query, dim_head)
    
    if self.variant == "cosine":
        # Cosine similarity: normalize then compute dot product
        proto_norm = F.normalize(proto_features, p=2, dim=1)  # (n_way, dim_head)
        query_norm = F.normalize(query_features, p=2, dim=1)  # (n_way*n_query, dim_head)
        scores = torch.matmul(query_norm, proto_norm.t())  # (n_way*n_query, n_way) ✅
        scores = scores * 10.0  # Temperature scaling
    else:
        # Euclidean distance (negative, so higher is better)
        scores = -torch.cdist(query_features, proto_features, p=2)  # (n_way*n_query, n_way) ✅
    
    return scores  # Correct shape: (n_way*n_query, n_way) ✅
```

### Fix 3: Add Missing Parameters
Added all missing CLI parameters to `io_utils.py`:
- `--gradient_accumulation_steps` (default: 2)
- `--use_amp` (default: 1)
- VIC-related parameters with their documented defaults

### Fix 4: Use Parameters from Config
Modified training function to use `params.gradient_accumulation_steps` and `params.use_amp` instead of hardcoded values.

### Fix 5: Make wandb Optional
Made wandb import optional to allow testing without requiring wandb installation.

## Expected Improvements

### Immediate Impact
With the critical bug fix, the model should now:

1. **Actually learn** - Each query gets its own score vector
2. **Achieve >20% accuracy** - No longer stuck at random guessing
3. **Have reasonable loss values** - Should be in range of 1-10, not 5000+
4. **Show diverse predictions** - Confusion matrix should show predictions across all classes

### Expected Accuracy Range
Based on the fixes:

- **Minimum expected**: 35-40% (basic learning should occur)
- **Target improvement**: >31% (achieving the requested 10% improvement)
- **Optimistic**: 45-55% (if all components work well together)

The actual performance will depend on:
- Dataset quality and preprocessing
- Hyperparameter tuning
- Number of training epochs
- Model convergence

## How to Test

Run the original command:
```bash
python train_test.py --method ProFOCT_cosine --gradient_accumulation_steps 2 \
    --dataset miniImagenet --backbone ResNet34 --FETI 1 --n_way 5 --k_shot 1 \
    --train_aug 0 --n_episode 2 --test_iter 2
```

With proper dataset setup (requires running `write_miniImagenet_filelist.py` after extracting dataset), you should see:

1. **Training accuracy increasing** over epochs (not stuck at 20%)
2. **Validation accuracy >30%** (achieving 10%+ improvement)
3. **Loss values decreasing** and staying in reasonable range (1-20)
4. **Confusion matrix showing predictions across multiple classes**

## Summary

The main issue was a fundamental architectural bug where the model was not computing individual scores for each query. This has been fixed, along with adding support for the ProFOCT method name and missing CLI parameters. The model should now be able to learn properly and achieve significantly better accuracy than the random-guessing baseline.
