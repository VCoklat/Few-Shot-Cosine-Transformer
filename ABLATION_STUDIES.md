# Ablation Studies Guide

This guide explains how to perform ablation studies to analyze the contribution of each component in the Few-Shot Cosine Transformer model.

## Overview

Ablation studies help understand which components contribute most to model performance. The following ablation configurations are recommended:

## Ablation Study Configurations

### 1. Model without SE Blocks (Channel Attention)

To measure the contribution of Squeeze-and-Excitation (SE) blocks for channel attention:

**Configuration:**
- Disable SE blocks in the backbone architecture
- Keep all other components active

**How to run:**
Modify the backbone initialization in your model to set `use_se=False` if the parameter exists, or use a backbone variant without SE blocks.

```python
# In backbone.py or your model configuration
# Set SE block usage to False
```

### 2. Model without Cosine Attention

To measure the impact of cosine attention vs. standard dot-product attention:

**Configuration:**
- Use `FSCT_softmax` or `CTX_softmax` instead of `FSCT_cosine` or `CTX_cosine`

**Command:**
```bash
python test.py --method FSCT_softmax --dataset miniImagenet --n_way 5 --k_shot 5
```

### 3. Model without VIC Regularization

To measure the contribution of Variance-Invariance-Covariance regularization:

**Configuration:**
- Disable VIC loss terms during training
- Set VIC regularization weight to 0

**How to run:**
Modify the training script to disable VIC loss:
```python
# In your training loop
vic_loss_weight = 0.0
```

### 4. Model without Dynamic Weighting

To evaluate the impact of dynamic weight adjustment:

**Configuration:**
- Use fixed weights instead of learned/adaptive weights
- Set dynamic weighting parameters to constant values

**How to run:**
Modify model to use fixed prototype weights or attention weights.

### 5. Model with Single VIC Component

To analyze individual VIC components:

**Variance Only:**
- Enable variance regularization only
- Disable invariance and covariance terms

**Covariance Only:**
- Enable covariance regularization only
- Disable variance and invariance terms

**How to run:**
Modify the VIC loss computation to include only selected terms.

### 6. Varying Number of Attention Heads

To find the optimal number of attention heads:

**Configurations to test:**
- 1 head
- 2 heads
- 4 heads (recommended optimal)
- 8 heads

**Command:**
```bash
# Modify the model initialization
# In methods/transformer.py or methods/CTX.py
# Change the 'heads' parameter: heads=1, heads=2, heads=4, heads=8
```

## Running Ablation Studies

### Basic Procedure

1. **Establish Baseline:**
   ```bash
   python train_test.py --method FSCT_cosine --dataset miniImagenet --n_way 5 --k_shot 5
   ```

2. **Test Each Ablation:**
   - Modify the configuration as described above
   - Train and evaluate with same settings as baseline
   - Record results

3. **Compare Results:**
   - Compare accuracy, F1-scores, and other metrics
   - Analyze performance drop for each removed component
   - Identify most critical components

### Recommended Testing Protocol

For each ablation configuration:
- Use the same dataset split
- Use the same random seed for reproducibility
- Run with same number of episodes (e.g., 600 for testing)
- Use comprehensive evaluation for detailed metrics

```bash
python test.py --method FSCT_cosine \
               --dataset miniImagenet \
               --n_way 5 \
               --k_shot 5 \
               --comprehensive_eval 1 \
               --feature_analysis 1
```

## Analyzing Results

### Key Metrics to Compare

1. **Accuracy:** Overall classification accuracy
2. **95% Confidence Interval:** Uncertainty in performance estimates
3. **Per-Class F1-Score:** Performance on individual classes
4. **Confusion Matrix:** Error patterns
5. **Feature Quality Metrics:**
   - Feature collapse rate
   - Feature utilization
   - Redundancy levels
   - Intra-class consistency

### Performance Drop Analysis

Calculate the performance drop for each ablation:

```
Performance Drop = (Baseline Accuracy - Ablation Accuracy) / Baseline Accuracy * 100%
```

This indicates the contribution of each component.

### Example Results Table

| Configuration | Accuracy | Δ from Baseline | F1-Score | Key Observations |
|--------------|----------|-----------------|----------|------------------|
| Full Model (Baseline) | 73.42% | - | 0.7340 | - |
| w/o SE Blocks | 71.85% | -1.57% | 0.7180 | Minor drop in channel adaptation |
| w/o Cosine Attention | 70.23% | -3.19% | 0.7020 | Significant impact on stability |
| w/o VIC Regularization | 72.10% | -1.32% | 0.7205 | Affects feature quality |
| w/o Dynamic Weighting | 71.50% | -1.92% | 0.7145 | Reduces adaptation capability |
| 1 Head | 69.80% | -3.62% | 0.6975 | Insufficient attention capacity |
| 2 Heads | 71.20% | -2.22% | 0.7115 | Better but still suboptimal |
| 8 Heads | 72.90% | -0.52% | 0.7285 | Slight overfitting risk |

## Advanced Analysis

### Feature Space Analysis

When running ablation studies with `--feature_analysis 1`, pay attention to:

1. **Feature Collapse:** Does removing a component cause more feature collapse?
2. **Redundancy:** Does the ablation increase feature redundancy?
3. **Consistency:** How does intra-class consistency change?
4. **Class Separability:** Check confusing class pairs

### Statistical Significance

Ensure differences are statistically significant:
- Use confidence intervals
- Run multiple seeds
- Apply statistical tests (t-test, ANOVA)

## Tips for Ablation Studies

1. **Keep Everything Else Constant:** Only change one component at a time
2. **Document Changes:** Keep detailed notes of modifications
3. **Use Version Control:** Git branch for each ablation
4. **Monitor Training:** Check if training dynamics change
5. **Resource Planning:** Some ablations may require different computational resources

## Implementation Notes

### Code Modifications

Most ablation studies require minimal code changes:

1. **Attention Type:** Use `--method` flag
2. **Hyperparameters:** Modify in model initialization
3. **Loss Components:** Comment out in training loop
4. **Architecture Changes:** Modify model definitions

### Preserving Results

Save results systematically:

```bash
# Create directory for ablation results
mkdir -p ablation_results

# Save each run with descriptive name
python test.py --method FSCT_softmax ... > ablation_results/no_cosine_attention.txt
```

## Conclusion

Ablation studies provide crucial insights into model design choices. By systematically removing or modifying components, you can:

- Identify critical components
- Guide future improvements
- Justify design decisions
- Optimize model complexity

Remember to run comprehensive evaluations for each ablation to get complete understanding of component contributions.
