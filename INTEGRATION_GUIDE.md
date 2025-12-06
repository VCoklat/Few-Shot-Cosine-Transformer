# Integration Guide: Enhanced Few-Shot Model

This guide explains how to integrate the Enhanced Few-Shot model with the existing training infrastructure.

## Quick Start

### Option 1: Using the New Training Script (Recommended)

The easiest way to use the enhanced model is with the dedicated training script:

```bash
# Train on miniImageNet 1-shot with Conv4
python train_enhanced.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 1 \
    --n_query 16 \
    --num_epoch 100 \
    --learning_rate 1e-3
```

### Option 2: Integrating with Existing train.py

To integrate the enhanced model with the existing `train.py`, follow these steps:

#### Step 1: Update io_utils.py

Add the enhanced model method to the argument parser:

```python
parser.add_argument('--method', default='FSCT_cosine',
                   help='CTX_softmax/CTX_cosine/FSCT_softmax/FSCT_cosine/OptimalFewShot/EnhancedOptimalFewShot')
```

#### Step 2: Modify train.py

Add the import at the top of `train.py`:

```python
from models.optimal_fewshot_enhanced import get_model_for_dataset
from methods.optimal_few_shot import OptimalFewShotModel
```

Add a new method handler in the training section (around line 203):

```python
elif params.method == 'EnhancedOptimalFewShot':
    few_shot_params = dict(
        n_way=params.n_way, k_shot=params.k_shot, n_query=params.n_query)
    
    base_datamgr = SetDataManager(
        image_size, n_episode=params.n_episode, **few_shot_params)
    base_loader = base_datamgr.get_data_loader(
        base_file, aug=params.train_aug)

    val_datamgr = SetDataManager(
        image_size, n_episode=params.n_episode, **few_shot_params)
    val_loader = val_datamgr.get_data_loader(
        val_file, aug=False)
    
    seed_func()
    
    def feature_model():
        if params.dataset in ['Omniglot', 'cross_char']:
            params.backbone = change_model(params.backbone)
        return model_dict[params.backbone](params.FETI, params.dataset) \
            if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset)
    
    # Use factory function for automatic dataset-specific configuration
    model = get_model_for_dataset(
        dataset=params.dataset,
        model_func=feature_model,
        n_way=params.n_way,
        k_shot=params.k_shot,
        n_query=params.n_query,
        feature_dim=64,  # Can be made configurable
        n_heads=4,
        dropout=0.1
    )
```

#### Step 3: Update test.py

Similarly, add support in `test.py` for testing the enhanced model:

```python
from models.optimal_fewshot_enhanced import get_model_for_dataset

# In the model loading section, add:
elif params.method == 'EnhancedOptimalFewShot':
    def feature_model():
        if params.dataset in ['Omniglot', 'cross_char']:
            params.backbone = change_model(params.backbone)
        return model_dict[params.backbone](params.FETI, params.dataset) \
            if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset)
    
    model = get_model_for_dataset(
        dataset=params.dataset,
        model_func=feature_model,
        n_way=params.n_way,
        k_shot=params.k_shot,
        n_query=params.n_query
    )
```

## Dataset-Specific Usage

### miniImageNet

```bash
# 1-shot with ResNet34 (target: +30.21% improvement)
python train_enhanced.py \
    --dataset miniImagenet \
    --backbone ResNet34 \
    --n_way 5 \
    --k_shot 1 \
    --n_query 16 \
    --learning_rate 5e-4 \
    --num_epoch 100

# 5-shot with Conv4
python train_enhanced.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 5 \
    --n_query 16 \
    --learning_rate 1e-3 \
    --num_epoch 100
```

### HAM10000 (Medical Imaging)

The model automatically uses medical-specific invariance for HAM10000:

```bash
# 1-shot (target: +0.36-0.47% improvement)
python train_enhanced.py \
    --dataset HAM10000 \
    --backbone Conv4 \
    --n_way 7 \
    --k_shot 1 \
    --n_query 16 \
    --learning_rate 1e-3 \
    --dropout 0.2 \
    --num_epoch 100

# 5-shot (target: +2.55% improvement for Conv4)
python train_enhanced.py \
    --dataset HAM10000 \
    --backbone Conv4 \
    --n_way 7 \
    --k_shot 5 \
    --n_query 16 \
    --learning_rate 1e-3 \
    --dropout 0.2 \
    --num_epoch 100
```

### CUB (Fine-grained Recognition)

The model automatically uses multi-scale invariance for CUB:

```bash
# 1-shot (target: +0.56-1.10% improvement)
python train_enhanced.py \
    --dataset CUB \
    --backbone ResNet18 \
    --n_way 5 \
    --k_shot 1 \
    --n_query 16 \
    --learning_rate 5e-4 \
    --dropout 0.15 \
    --num_epoch 100

# 5-shot
python train_enhanced.py \
    --dataset CUB \
    --backbone ResNet18 \
    --n_way 5 \
    --k_shot 5 \
    --n_query 16 \
    --learning_rate 5e-4 \
    --dropout 0.15 \
    --num_epoch 100
```

### Omniglot

```bash
# 1-shot
python train_enhanced.py \
    --dataset Omniglot \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 1 \
    --n_query 15 \
    --learning_rate 1e-3 \
    --dropout 0.05 \
    --num_epoch 50

# 5-shot
python train_enhanced.py \
    --dataset Omniglot \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 5 \
    --n_query 15 \
    --learning_rate 1e-3 \
    --dropout 0.05 \
    --num_epoch 50
```

## Testing

After training, test your model:

```bash
# Using the existing test.py (after integration)
python test.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --method EnhancedOptimalFewShot \
    --n_way 5 \
    --k_shot 1 \
    --test_iter 600
```

## Advanced Configuration

### Custom Module Configuration

To disable specific modules for ablation studies:

```python
from models.optimal_fewshot_enhanced import EnhancedOptimalFewShot

model = EnhancedOptimalFewShot(
    model_func=model_func,
    n_way=5,
    k_shot=1,
    n_query=16,
    use_task_invariance=True,       # Enable/disable
    use_multi_scale=False,           # Enable/disable
    use_feature_augmentation=True,   # Enable/disable
    use_prototype_refinement=False,  # Enable/disable
    domain='general'                 # 'general', 'medical', or 'fine_grained'
)
```

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **Learning Rate**:
   - Backbone: `lr * 0.1` (lower to preserve pretrained features)
   - Invariance modules: `lr` (higher to learn new transformations)

2. **Dropout**:
   - Omniglot: 0.05
   - miniImageNet: 0.1
   - CUB: 0.15
   - HAM10000: 0.2

3. **Feature Dimension**:
   - Default: 64
   - For better performance (more memory): 128

4. **Gradient Clipping**:
   - Default: 1.0
   - Increase if training is unstable

## Monitoring Training

The enhanced training script provides:

- Per-epoch training loss and accuracy
- Validation accuracy with confidence intervals
- Learning rate schedule
- Best model checkpointing

## Memory Considerations

The enhanced model uses:
- Gradient checkpointing for memory efficiency
- Approximately 10-20% more parameters than base model
- Should fit comfortably on 8GB GPU for Conv4 backbone
- May require gradient accumulation for ResNet34 with large feature dims

## Troubleshooting

### Issue: Out of Memory

**Solutions:**
1. Reduce feature dimension: `--feature_dim 32`
2. Reduce number of heads: `--n_heads 2`
3. Use smaller backbone (Conv4 instead of ResNet)
4. Reduce batch size (n_query)

### Issue: Training Unstable

**Solutions:**
1. Increase gradient clipping: `--grad_clip 0.5`
2. Reduce learning rate: `--learning_rate 5e-4`
3. Increase dropout: `--dropout 0.15`

### Issue: Slow Convergence

**Solutions:**
1. Increase learning rate for invariance modules
2. Reduce dropout
3. Use data augmentation: `--train_aug 1`

## Comparison with Base Model

To compare with the base OptimalFewShot model:

```bash
# Base model (using train.py)
python train.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --method OptimalFewShot \
    --n_way 5 \
    --k_shot 1

# Enhanced model
python train_enhanced.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 1
```

## Expected Training Time

Approximate training times on a single GPU:

- Omniglot: ~30 minutes (50 epochs)
- miniImageNet: ~4-6 hours (100 epochs)
- CUB: ~4-6 hours (100 epochs)
- HAM10000: ~3-5 hours (100 epochs)

Times may vary based on GPU type and configuration.

## Citation

When using this enhanced model, please cite both the original Few-Shot-Cosine-Transformer repository and acknowledge the enhancement modules.
