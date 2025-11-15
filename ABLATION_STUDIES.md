# Ablation Studies for Few-Shot Cosine Transformer

This document describes the ablation study configurations for analyzing component contributions in the Few-Shot Cosine Transformer (FSCT) model.

## Overview

Ablation studies systematically evaluate the impact of individual components by removing or modifying them. These studies help understand:
- Which components are essential for model performance
- How different design choices affect feature quality
- The contribution of each architectural element

## Baseline Configuration

The baseline FSCT_cosine model includes:
- **Backbone**: Conv4 or ResNet-based feature extractor
- **Cosine Attention**: Cosine similarity-based attention mechanism
- **SE Blocks**: Squeeze-and-Excitation blocks for channel attention
- **VIC Regularization**: Variance-Invariance-Covariance regularization
- **Dynamic Weighting**: Adaptive weighting of support samples
- **Multi-head Attention**: 4 attention heads (default)

## Ablation Configurations

### 1. Without SE Blocks (Channel Attention)

**Configuration:**
- Remove Squeeze-and-Excitation blocks from the backbone
- Keep all other components intact

**Purpose:** 
Evaluate the contribution of channel attention mechanism to feature quality and classification performance.

**Expected Impact:**
- May reduce feature utilization
- Could affect feature diversity
- Potential decrease in overall accuracy

**Command:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 50 \
    --test_iter 600 --feature_analysis 1 --se_blocks 0
```

---

### 2. Without Cosine Attention (Dot-Product Baseline)

**Configuration:**
- Replace cosine similarity attention with standard dot-product attention
- Use FSCT_softmax variant instead

**Purpose:**
Compare cosine-based attention against traditional softmax attention to understand the impact of normalized similarity measures.

**Expected Impact:**
- May increase feature redundancy
- Could affect intra-class consistency
- Different confusion pair patterns

**Command:**
```bash
python train_test.py --method FSCT_softmax --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 50 \
    --test_iter 600 --feature_analysis 1
```

---

### 3. Without VIC Regularization

**Configuration:**
- Disable Variance-Invariance-Covariance regularization
- Keep all other components

**Purpose:**
Assess the impact of VIC regularization on feature collapse, redundancy, and overall feature quality.

**Expected Impact:**
- Potential increase in feature collapse
- May reduce feature diversity
- Could affect effective dimensionality

**Command:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 50 \
    --test_iter 600 --feature_analysis 1 --vic_reg 0
```

---

### 4. Without Dynamic Weighting

**Configuration:**
- Use uniform weighting for support samples
- Disable adaptive importance weighting

**Purpose:**
Evaluate the contribution of dynamic weighting to handling class imbalance and support sample quality variation.

**Expected Impact:**
- May affect performance on imbalanced episodes
- Could impact confusing pair separation
- Potential changes in statistical confidence

**Command:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 50 \
    --test_iter 600 --feature_analysis 1 --dynamic_weighting 0
```

---

### 5. Single VIC Component (Variance Only)

**Configuration:**
- Use only variance component of VIC regularization
- Disable invariance and covariance terms

**Purpose:**
Isolate the contribution of variance regularization in preventing feature collapse.

**Expected Impact:**
- Partial protection against feature collapse
- May not address redundancy issues
- Intermediate performance between full VIC and no VIC

**Command:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 50 \
    --test_iter 600 --feature_analysis 1 --vic_variance_only 1
```

---

### 6. Varying Attention Heads

**Configuration:**
Test with different numbers of attention heads: 1, 2, 4, 8

**Purpose:**
Understand the relationship between multi-head attention capacity and feature quality metrics.

**Expected Impact:**
- More heads may improve feature diversity
- Could affect feature redundancy patterns
- Trade-off between expressiveness and efficiency

**Commands:**

**1 Head:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 50 \
    --test_iter 600 --feature_analysis 1 --n_heads 1
```

**2 Heads:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 50 \
    --test_iter 600 --feature_analysis 1 --n_heads 2
```

**4 Heads (Default):**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 50 \
    --test_iter 600 --feature_analysis 1 --n_heads 4
```

**8 Heads:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 50 \
    --test_iter 600 --feature_analysis 1 --n_heads 8
```

---

## Quick Test Run (Reduced Settings)

For rapid testing and validation, use these reduced settings:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --FETI 0 --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 \
    --num_epoch 1 --test_iter 600 --feature_analysis 1
```

This configuration:
- Uses only 1 training epoch (for quick testing)
- 200 episodes per epoch
- 600 test episodes for statistical significance
- Feature analysis enabled

---

## Feature Metrics Analyzed

For each ablation configuration, the following 8 feature metrics are automatically computed and displayed:

1. **Feature Collapse Detection**
   - Number and percentage of collapsed dimensions (std < 1e-4)
   - Min/mean/max standard deviations

2. **Feature Utilization**
   - Mean, median, min, max utilization
   - Percentile-based range vs theoretical maximum

3. **Feature Diversity**
   - Coefficient of variation from class centroids
   - Number of classes evaluated

4. **Feature Redundancy**
   - High correlation pairs (>0.9) and medium pairs (>0.7)
   - Effective dimensions at 95% variance (PCA)
   - Dimensionality reduction ratio

5. **Intra-class Consistency**
   - Mean and std of Euclidean distances to class centroids
   - Mean and std of cosine similarities

6. **Confusing Pairs**
   - Top confusing class pairs by centroid proximity
   - Inter-centroid distance statistics

7. **Class Imbalance**
   - Imbalance ratio (minority/majority)
   - Sample count statistics per class

8. **Statistical Confidence**
   - 95% confidence intervals from episode accuracies
   - Per-class F1 scores
   - Confusion matrix

---

## Interpreting Results

### Good Feature Space Characteristics:
- **Low collapse ratio** (< 5% of dimensions)
- **High utilization** (> 0.7 mean)
- **High diversity** (CV > 0.5)
- **Low redundancy** (few high-correlation pairs, effective dims close to total)
- **High intra-class consistency** (low distance variance, high cosine similarity)
- **Well-separated classes** (high inter-centroid distances)
- **Balanced classes** (imbalance ratio close to 1.0)
- **Tight confidence intervals** (low variance across episodes)

### Warning Signs:
- **High collapse ratio** indicates feature space degradation
- **Low utilization** suggests underutilization of feature capacity
- **High correlation pairs** indicate redundant features
- **Low effective dimensions** relative to total suggests information compression
- **Confusing pairs with very small distances** indicate difficult class separations

---

## Running All Ablations

To systematically run all ablation studies, create a script:

```bash
#!/bin/bash

# Array of configurations
configs=(
    "--method FSCT_cosine --se_blocks 0"  # No SE blocks
    "--method FSCT_softmax"  # Dot-product attention
    "--method FSCT_cosine --vic_reg 0"  # No VIC
    "--method FSCT_cosine --dynamic_weighting 0"  # No dynamic weighting
    "--method FSCT_cosine --vic_variance_only 1"  # Variance only
    "--method FSCT_cosine --n_heads 1"  # 1 head
    "--method FSCT_cosine --n_heads 2"  # 2 heads
    "--method FSCT_cosine --n_heads 8"  # 8 heads
)

# Base command
base="python train_test.py --dataset miniImagenet --backbone Conv4 --FETI 0 \
      --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 --num_epoch 1 \
      --test_iter 600 --feature_analysis 1"

# Run each configuration
for config in "${configs[@]}"; do
    echo "Running: $config"
    $base $config
    echo "---"
done
```

---

## Notes

- **Note 1:** Some flags (--se_blocks, --vic_reg, --dynamic_weighting, --vic_variance_only, --n_heads) may need to be added to the model implementation if not already present. The commands above assume these flags exist or can be easily added.

- **Note 2:** Feature analysis automatically extracts features from the model during evaluation. Ensure your model has a `.feature` attribute that can be called for feature extraction.

- **Note 3:** For publication-quality results, use more training epochs (e.g., --num_epoch 50) and more test episodes (e.g., --test_iter 1000).

- **Note 4:** Statistical significance requires multiple runs with different random seeds. Consider running each configuration 3-5 times and reporting mean ± std.

---

## Expected Output

Each ablation run will produce comprehensive output including:

1. **Standard Classification Metrics**
   - Accuracy, F1 scores, precision, recall
   - Confusion matrix
   - Per-class performance

2. **Statistical Confidence**
   - 95% confidence intervals
   - Episode-wise performance distribution

3. **8 Feature Analysis Metrics**
   - Detailed breakdown of feature space quality
   - Component-wise analysis
   - Visual indicators of feature health

4. **Hardware Utilization**
   - GPU/CPU usage
   - Memory consumption
   - Inference time per episode

This comprehensive analysis enables deep understanding of how each component contributes to both traditional accuracy metrics and underlying feature space quality.
