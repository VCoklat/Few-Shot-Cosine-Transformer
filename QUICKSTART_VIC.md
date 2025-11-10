# VIC-Enhanced FS-CT: Quick Start Guide

## What is VIC Loss?

VIC (Variance-Invariance-Covariance) loss is an enhancement to the Few-Shot Cosine Transformer (FS-CT) training process that adds regularization to improve model performance and feature quality.

## Quick Start

### Basic Training (Original Method)
```bash
python train_test.py --method FSCT_cosine \
                     --dataset miniImagenet \
                     --backbone ResNet34 \
                     --n_way 5 \
                     --k_shot 5
```

### VIC-Enhanced Training (Recommended)
```bash
python train_test.py --method FSCT_cosine \
                     --dataset miniImagenet \
                     --backbone ResNet34 \
                     --n_way 5 \
                     --k_shot 5 \
                     --lambda_I 1.0 \
                     --lambda_V 0.5 \
                     --lambda_C 0.1
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda_I` | 1.0 | Weight for Invariance Loss (cross-entropy) |
| `--lambda_V` | 0.0 | Weight for Variance Loss (compactness) |
| `--lambda_C` | 0.0 | Weight for Covariance Loss (decorrelation) |

## Loss Components

### L_I: Invariance Loss
- **What**: Standard cross-entropy loss
- **Purpose**: Ensures predictions match ground truth
- **When to use**: Always (default weight = 1.0)

### L_V: Variance Loss
- **What**: Hinge loss on standard deviation
- **Purpose**: Encourages compact support embeddings
- **When to use**: When support samples are scattered
- **Recommended range**: 0.1 - 1.0

### L_C: Covariance Loss
- **What**: Covariance regularization
- **Purpose**: Decorrelates feature dimensions
- **When to use**: To prevent feature collapse
- **Recommended range**: 0.01 - 0.5

## Common Configurations

### Configuration 1: Standard Training
```bash
--lambda_I 1.0 --lambda_V 0.0 --lambda_C 0.0
```
- Original FS-CT behavior
- No additional regularization
- Fastest training

### Configuration 2: With Compactness
```bash
--lambda_I 1.0 --lambda_V 0.5 --lambda_C 0.0
```
- Encourages compact class representations
- Good for datasets with high intra-class variance

### Configuration 3: With Decorrelation
```bash
--lambda_I 1.0 --lambda_V 0.0 --lambda_C 0.1
```
- Prevents feature dimension correlation
- Good for preventing overfitting

### Configuration 4: Full VIC (Recommended)
```bash
--lambda_I 1.0 --lambda_V 0.5 --lambda_C 0.1
```
- Balances all loss components
- Best overall performance
- Recommended starting point

## Testing Your Implementation

Run the validation script to verify everything works:
```bash
python validate_algorithm.py
```

Run unit tests:
```bash
python test_vic_loss.py
```

Run integration tests:
```bash
python test_vic_integration.py
```

## Hyperparameter Tuning Tips

1. **Start with defaults**: Begin with λ_I=1.0, λ_V=0.5, λ_C=0.1
2. **Adjust variance weight**: Increase if embeddings are scattered, decrease if too compact
3. **Adjust covariance weight**: Increase if experiencing feature collapse
4. **Monitor training**: Watch both loss and accuracy - good balance is key
5. **Dataset-specific**: Adjust based on dataset characteristics

## Example Training Commands

### Mini-ImageNet (5-way 1-shot)
```bash
python train_test.py --method FSCT_cosine \
                     --dataset miniImagenet \
                     --backbone ResNet34 \
                     --n_way 5 --k_shot 1 \
                     --lambda_I 1.0 --lambda_V 0.3 --lambda_C 0.1
```

### Mini-ImageNet (5-way 5-shot)
```bash
python train_test.py --method FSCT_cosine \
                     --dataset miniImagenet \
                     --backbone ResNet34 \
                     --n_way 5 --k_shot 5 \
                     --lambda_I 1.0 --lambda_V 0.5 --lambda_C 0.1
```

### CUB (5-way 5-shot)
```bash
python train_test.py --method FSCT_cosine \
                     --dataset CUB \
                     --backbone ResNet34 \
                     --n_way 5 --k_shot 5 \
                     --lambda_I 1.0 --lambda_V 0.7 --lambda_C 0.15
```

## Troubleshooting

### Loss is too high
- Decrease λ_V and λ_C
- Focus on λ_I (invariance) first

### Poor accuracy
- Ensure λ_I is at least 1.0
- Try increasing λ_V for better compactness

### Training is unstable
- Reduce λ_V and λ_C by 50%
- Lower learning rate

### Features are collapsing
- Increase λ_C (covariance weight)
- Check if λ_V is too high

## FAQ

**Q: Do I need to use VIC loss?**  
A: No, it's optional. Setting λ_V=0 and λ_C=0 gives you the original FS-CT.

**Q: What's the best configuration?**  
A: Start with λ_I=1.0, λ_V=0.5, λ_C=0.1 and tune from there.

**Q: Does VIC loss slow down training?**  
A: Minimal impact - the additional computation is negligible.

**Q: Can I use VIC with other methods?**  
A: Currently implemented for FSCT_cosine and FSCT_softmax only.

**Q: How do I know if VIC is helping?**  
A: Compare validation accuracy with and without VIC loss.

## Support

For more details, see:
- `VIC_IMPLEMENTATION.md` - Technical documentation
- `README.md` - Main repository documentation
- `methods/transformer.py` - Implementation code

For issues or questions, please refer to the repository's issue tracker.
