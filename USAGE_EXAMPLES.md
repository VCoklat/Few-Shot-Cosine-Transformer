# Usage Examples for Feature Analysis and Ablation Studies

This document provides practical examples for running feature analysis and ablation studies with the Few-Shot Cosine Transformer.

## Quick Start

### Basic Feature Analysis

Run a quick test with comprehensive feature analysis:

```bash
python train_test.py \
  --method FSCT_cosine \
  --dataset miniImagenet \
  --backbone Conv4 \
  --FETI 0 \
  --n_way 5 \
  --k_shot 1 \
  --train_aug 0 \
  --n_episode 200 \
  --num_epoch 1 \
  --test_iter 600 \
  --feature_analysis 1
```

This will output:
- Standard classification metrics (accuracy, F1, precision, recall)
- Statistical confidence (95% CI from episode accuracies)
- All 8 feature space metrics:
  1. Feature Collapse Detection
  2. Feature Utilization
  3. Feature Diversity
  4. Feature Redundancy
  5. Intra-class Consistency
  6. Confusing Pairs
  7. Class Imbalance
  8. Statistical Confidence

## Ablation Studies

### 1. Varying Number of Attention Heads

Test with different numbers of attention heads:

**1 Head:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1 --n_heads 1
```

**2 Heads:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1 --n_heads 2
```

**4 Heads (Default):**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1 --n_heads 4
```

**8 Heads:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1 --n_heads 8
```

### 2. Without SE Blocks

Test the model without Squeeze-and-Excitation blocks:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1 --se_blocks 0
```

### 3. Cosine vs Softmax Attention

Compare cosine attention against softmax (dot-product) attention:

**Cosine Attention:**
```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1
```

**Softmax Attention:**
```bash
python train_test.py --method FSCT_softmax --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1
```

### 4. Without VIC Regularization

Test without Variance-Invariance-Covariance regularization:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1 --vic_reg 0
```

### 5. VIC Variance Only

Use only the variance component of VIC regularization:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1 --vic_variance_only 1
```

### 6. Without Dynamic Weighting

Disable dynamic weighting for support samples:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1 --dynamic_weighting 0
```

## Understanding the Output

When `--feature_analysis 1` is enabled, you'll see comprehensive output including:

### Classification Metrics Section
```
📊 Classification Metrics:
  Accuracy:          0.6523
  Macro-F1:          0.6489
  Macro Precision:   0.6612
  Macro Recall:      0.6523
  Cohen's κ:         0.5654
  Matthews CorrCoef: 0.5663
  Top-5 Accuracy:    1.0000
```

### Statistical Confidence Section
```
📊 Statistical Confidence (95% CI):
  Mean Episode Accuracy: 65.23%
  Standard Deviation:    2.34%
  95% Confidence Interval: [62.89%, 67.57%]
  Number of Episodes:    600
```

### Feature Analysis Section
```
============================================================
COMPREHENSIVE FEATURE SPACE ANALYSIS
============================================================

📉 Feature Collapse:
  Collapsed dimensions: 0/64 (0.0%)
  Min/Mean/Max std: 0.234567 / 1.234567 / 3.456789

📊 Feature Utilization:
  Mean: 0.7234
  Median: 0.7189
  Range: [0.4523, 0.9012]

🎨 Feature Diversity:
  Coefficient of Variation: 0.6123
  Num classes: 5

🔄 Feature Redundancy:
  High correlation pairs (>0.9): 2
  Medium correlation pairs (>0.7): 8
  Effective dims (95% variance): 47/64

🎯 Intra-class Consistency:
  Mean Euclidean distance: 2.3456 ± 0.4567
  Mean Cosine similarity: 0.8234 ± 0.1234

🤔 Most Confusing Class Pairs:
  1. Classes 2 ↔ 3: distance = 1.2345
  2. Classes 0 ↔ 4: distance = 1.3456
  3. Classes 1 ↔ 2: distance = 1.4567

⚖️ Class Imbalance:
  Imbalance ratio: 1.0000
  Samples per class: 16 - 16 (mean: 16.0)

📈 Statistical Confidence:
  Mean accuracy: 65.23%
  95% CI: [62.89%, 67.57%]
  Macro F1: 0.6489
```

## Advanced Usage

### Disable Feature Analysis

If you only want standard metrics without feature analysis:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 0 --comprehensive_eval 0
```

### Combine Multiple Ablations

Test multiple ablations simultaneously:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 1 --n_episode 200 --num_epoch 1 --test_iter 600 \
  --feature_analysis 1 --n_heads 2 --se_blocks 0 --dynamic_weighting 0
```

### Full Training Run

For production runs, use more epochs and test iterations:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
  --n_way 5 --k_shot 5 --train_aug 1 --n_episode 600 --num_epoch 50 \
  --test_iter 1000 --feature_analysis 1 --comprehensive_eval 1
```

## Available Arguments

### Core Arguments
- `--method`: Model method (FSCT_cosine, FSCT_softmax, CTX_cosine, CTX_softmax)
- `--dataset`: Dataset (miniImagenet, CIFAR, CUB, Omniglot)
- `--backbone`: Feature extractor (Conv4, Conv6, ResNet12, ResNet18)
- `--n_way`: Number of classes per episode
- `--k_shot`: Number of support samples per class
- `--n_query`: Number of query samples per class
- `--n_episode`: Episodes per epoch
- `--num_epoch`: Total training epochs
- `--test_iter`: Number of test episodes

### Evaluation Arguments
- `--comprehensive_eval`: Enable comprehensive evaluation (default: 1)
- `--feature_analysis`: Enable feature space analysis (default: 1)

### Ablation Study Arguments
- `--se_blocks`: Use Squeeze-and-Excitation blocks (default: 1)
- `--n_heads`: Number of attention heads (default: 4)
- `--vic_reg`: Use VIC regularization (default: 1)
- `--vic_variance_only`: Use only variance component of VIC (default: 0)
- `--dynamic_weighting`: Use dynamic support sample weighting (default: 1)

## Tips

1. **Quick Testing**: Use `--num_epoch 1` and `--test_iter 200` for rapid experimentation
2. **Statistical Significance**: Use `--test_iter 600` or higher for reliable confidence intervals
3. **Feature Analysis**: Always enable for understanding model behavior
4. **Compare Ablations**: Run baseline first, then ablations one at a time to isolate effects
5. **Multiple Runs**: For publication, run each configuration 3-5 times with different seeds

## Interpreting Results

### Good Feature Space Signs
- Collapse ratio < 5%
- Utilization > 0.7
- Diversity CV > 0.5
- Low redundancy (few high-correlation pairs)
- High intra-class consistency (high cosine similarity)
- Large inter-centroid distances

### Warning Signs
- High collapse ratio indicates feature degradation
- Low utilization suggests underused capacity
- Many high-correlation pairs indicate redundancy
- Low effective dimensionality suggests information loss
- Small inter-centroid distances indicate confusable classes
