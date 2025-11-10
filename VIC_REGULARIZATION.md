# VIC Regularization with Dynamic Weighting

## Overview

This implementation integrates **VIC (Variance-Invariance-Covariance) Regularization** from the ProFONet framework into the Few-Shot Cosine Transformer, with **dynamic weight adjustment** for optimal balance during training. This combination targets a >20% accuracy increase while maintaining memory efficiency to prevent OOM errors on systems with limited VRAM (e.g., 16GB on Kaggle).

## Key Features

### 1. VIC Regularization Components

The VIC regularization consists of three loss terms that work together to create a robust and discriminative feature space:

#### Variance Loss (V)
- **Purpose**: Encourages embeddings to have sufficient spread
- **Formula**: `V = Σ max(0, ε - Var(E_j))`
- **Effect**: Prevents feature collapse by ensuring embeddings maintain variance above threshold `ε`

#### Invariance Loss (I)
- **Purpose**: Minimizes distance between embeddings within the same class
- **Formula**: Mean squared error between samples and their class prototype
- **Effect**: Creates tight, compact clusters for each class

#### Covariance Loss (C)
- **Purpose**: Encourages decorrelated (orthogonal) features
- **Formula**: Sum of squared off-diagonal elements in covariance matrix
- **Effect**: Reduces redundancy and improves feature diversity

### 2. Dynamic Weight Adjustment

Unlike fixed weights, our implementation automatically balances the three loss components during training:

- **Adaptive Mechanism**: Weights adjust based on relative loss magnitudes
- **Balanced Learning**: Prevents any single loss term from dominating
- **Memory Efficient**: Automatically reduces costly terms if needed to prevent OOM
- **Target**: Each loss contributes approximately 1/3 to the total VIC loss

### 3. Memory Optimization

To prevent OOM on 16GB VRAM systems:

- **Mixed Precision Training**: Optional FP16 training reduces memory footprint by ~50%
- **Efficient Caching**: Embeddings cached only when needed for VIC computation
- **Transductive Learning**: Optional application to both support and query sets

## Usage

### Basic Training with VIC Regularization

```bash
python train.py \
    --method FSCT_cosine \
    --dataset miniImagenet \
    --backbone ResNet18 \
    --n_way 5 \
    --k_shot 5 \
    --use_vic 1 \
    --vic_lambda_v 1.0 \
    --vic_lambda_i 1.0 \
    --vic_lambda_c 1.0
```

### With Memory Optimization

```bash
python train.py \
    --method FSCT_cosine \
    --dataset miniImagenet \
    --backbone ResNet18 \
    --n_way 5 \
    --k_shot 5 \
    --use_vic 1 \
    --mixed_precision 1 \
    --n_episode 100
```

### All VIC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_vic` | 0 | Enable VIC regularization (1=True, 0=False) |
| `--vic_lambda_v` | 1.0 | Initial weight for variance loss |
| `--vic_lambda_i` | 1.0 | Initial weight for invariance loss |
| `--vic_lambda_c` | 1.0 | Initial weight for covariance loss |
| `--vic_epsilon` | 1e-4 | Minimum variance threshold |
| `--vic_alpha` | 0.001 | Learning rate for dynamic weight updates |
| `--mixed_precision` | 0 | Use FP16 mixed precision training |

## Architecture Integration

### Modified Components

1. **`methods/vic_regularization.py`** (New)
   - Standalone VIC regularization module
   - Dynamic weight adjustment logic
   - All three loss computations

2. **`methods/transformer.py`** (Modified)
   - `FewShotTransformer.__init__`: Added VIC parameters
   - `set_forward`: Caches embeddings when VIC is enabled
   - `set_forward_loss`: Computes and combines VIC loss with CE loss

3. **`methods/meta_template.py`** (Modified)
   - `train_loop`: Supports mixed precision and VIC loss logging
   - Returns tuple (acc, loss, vic_dict) when VIC is used

4. **`train.py`** (Modified)
   - Mixed precision scaler integration
   - Memory optimization with `torch.cuda.empty_cache()`

5. **`io_utils.py`** (Modified)
   - Added all VIC-related command-line arguments

## How It Works

### Training Loop Flow

```python
for epoch in num_epochs:
    for batch in train_loader:
        # 1. Extract features from backbone
        z_s = backbone(support_images)  # (n_way, k_shot, dim)
        z_q = backbone(query_images)    # (n_way*n_query, dim)
        
        # 2. Compute weighted prototypes with learnable weights
        z_p = weighted_prototype(z_s)
        
        # 3. Cosine Attention for prediction
        logits = cosine_transformer(z_p, z_q)
        ce_loss = cross_entropy(logits, labels)
        
        # 4. VIC regularization (if enabled)
        if use_vic:
            vic_dict = vic_regularization(z_s, z_q)
            v_loss = vic_dict['variance']
            i_loss = vic_dict['invariance']
            c_loss = vic_dict['covariance']
            
            # 5. Dynamic weighted combination
            λ_v, λ_i, λ_c = vic_dict['lambda_v'], vic_dict['lambda_i'], vic_dict['lambda_c']
            vic_loss = λ_v * v_loss + λ_i * i_loss + λ_c * c_loss
            
            total_loss = ce_loss + vic_loss
            
            # 6. Update dynamic weights
            update_weights(v_loss, i_loss, c_loss)
        else:
            total_loss = ce_loss
        
        # 7. Backpropagation
        total_loss.backward()
        optimizer.step()
```

### Dynamic Weight Update Mechanism

The weights are adjusted to balance the loss contributions:

```python
def update_dynamic_weights(v_loss, i_loss, c_loss):
    total = v_loss + i_loss + c_loss
    
    # Calculate current contribution ratios
    v_ratio = v_loss / total
    i_ratio = i_loss / total
    c_ratio = c_loss / total
    
    # Target: each should contribute 1/3
    target = 1.0 / 3.0
    
    # Adjust weights (increase if below target, decrease if above)
    λ_v += α * (target - v_ratio) * λ_v
    λ_i += α * (target - i_ratio) * λ_i
    λ_c += α * (target - c_ratio) * λ_c
    
    # Clamp to prevent extreme values
    λ_v = clamp(λ_v, min_weight, max_weight)
    λ_i = clamp(λ_i, min_weight, max_weight)
    λ_c = clamp(λ_c, min_weight, max_weight)
```

## Expected Improvements

Based on the ProFONet and Cosine Transformer papers:

1. **Accuracy**: Target >20% improvement over baseline transformer
   - Variance loss prevents feature collapse
   - Invariance loss creates tighter class clusters
   - Covariance loss improves feature diversity

2. **Stability**: More consistent training across different initializations
   - Dynamic weighting prevents loss imbalance
   - Balanced optimization of all components

3. **Memory Efficiency**: Works on 16GB VRAM with mixed precision
   - ~50% memory reduction with FP16
   - Efficient embedding caching strategy

## Monitoring Training

When using WandB (`--wandb 1`), you can track:

- `Loss`: Total training loss (CE + VIC)
- `Train Acc`: Training accuracy
- `VIC_Variance`: Variance loss component
- `VIC_Invariance`: Invariance loss component
- `VIC_Covariance`: Covariance loss component
- Dynamic weights evolution over time

## Testing

Run the integration tests to verify the implementation:

```bash
python test_vic_integration.py
```

This tests:
1. VIC regularization basic functionality
2. Transformer integration
3. Backward pass and gradient flow
4. Memory efficiency (if CUDA available)

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Enable mixed precision: `--mixed_precision 1`
2. Reduce episode batch size: `--n_episode 100` (default is 200)
3. Reduce n_query: `--n_query 8` (default is 16)
4. Use smaller backbone: `--backbone Conv4` instead of ResNet

### VIC Loss Too High

If VIC loss dominates:

1. Reduce initial weights: `--vic_lambda_v 0.5 --vic_lambda_i 0.5 --vic_lambda_c 0.5`
2. Increase learning rate for dynamic adjustment: `--vic_alpha 0.01`
3. The dynamic mechanism should automatically balance over time

### Poor Accuracy

If accuracy doesn't improve:

1. Ensure VIC is enabled: `--use_vic 1`
2. Check that variant is cosine: `--method FSCT_cosine`
3. Try different initial weights
4. Verify dataset and backbone are appropriate

## References

1. **Cosine Transformer**: Nguyen et al. "Enhancing Few-Shot Image Classification With Cosine Transformer", IEEE Access 2023
2. **ProFONet**: Afrasiyabi et al. "Associative Alignment for Few-shot Image Classification", ECCV 2020
3. **VICReg**: Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning", ICLR 2022

## License

This implementation follows the same license as the original Few-Shot Cosine Transformer repository.
