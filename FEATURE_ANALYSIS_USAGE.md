# Feature Analysis Usage Guide

## Quick Start

Run comprehensive evaluation with all 8 feature metrics in one command:

```bash
python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 \
    --FETI 0 --n_way 5 --k_shot 1 --train_aug 0 --n_episode 200 \
    --num_epoch 1 --test_iter 600 --feature_analysis 1
```

## What Gets Computed

When `--feature_analysis 1` is set, the system automatically computes and displays:

### 1. Classification Metrics
- Accuracy, Precision, Recall, F1 scores
- Cohen's Kappa, Matthews Correlation Coefficient
- Top-5 Accuracy
- Confusion Matrix
- Per-class F1 scores

### 2. Statistical Confidence
- 95% confidence intervals from per-episode accuracies
- Z-score approximation
- Episode-wise performance distribution

### 3. Feature Space Analysis (8 Metrics)

#### 📉 Feature Collapse Detection
- Number of collapsed dimensions (std < 1e-4)
- Min/Mean/Max standard deviations
- Collapse ratio percentage

#### 📊 Feature Utilization
- Mean, median, min, max utilization
- Percentile-based range (5th to 95th percentile)
- Comparison to theoretical maximum

#### 🎨 Feature Diversity
- Coefficient of variation from class centroids
- Measures how well classes are separated in feature space

#### 🔄 Feature Redundancy
- High correlation pairs (>0.9)
- Medium correlation pairs (>0.7)
- PCA effective dimensionality at 95% variance
- Dimensionality reduction ratio

#### 🎯 Intra-class Consistency
- Mean Euclidean distance to class centroids
- Mean cosine similarity within classes
- Standard deviations for both metrics

#### 🤔 Confusing Class Pairs
- Top-k most confusing pairs based on centroid proximity
- Inter-centroid distance statistics

#### ⚖️ Class Imbalance
- Imbalance ratio (minority/majority)
- Sample count statistics per class

#### 📈 Statistical Confidence
- 95% CI from episode accuracies
- Per-class F1 breakdown
- Confusion matrix

## Example Output

```
============================================================
COMPREHENSIVE FEATURE SPACE ANALYSIS
============================================================

[1/8] Detecting feature collapse...
[2/8] Computing feature utilization...
[3/8] Analyzing feature diversity...
[4/8] Assessing feature redundancy...
[5/8] Evaluating intra-class consistency...
[6/8] Identifying confusing class pairs...
[7/8] Computing class imbalance...
[8/8] Calculating statistical confidence...

✓ Analysis complete!

============================================================
FEATURE ANALYSIS SUMMARY
============================================================

📉 Feature Collapse:
  Collapsed dimensions: 0/64 (0.0%)
  Min/Mean/Max std: 0.860942 / 0.992082 / 1.111071

📊 Feature Utilization:
  Mean: 0.6467
  Median: 0.6512
  Range: [0.5013, 0.8218]

🎨 Feature Diversity:
  Coefficient of Variation: 7.0193
  Num classes: 5

🔄 Feature Redundancy:
  High correlation pairs (>0.9): 0
  Medium correlation pairs (>0.7): 0
  Effective dims (95% variance): 47/64

🎯 Intra-class Consistency:
  Mean Euclidean distance: 7.7568 ± 0.6463
  Mean Cosine similarity: 0.2205 ± 0.1212

🤔 Most Confusing Class Pairs:
  1. Classes 2 ↔ 4: distance = 2.3315
  2. Classes 0 ↔ 1: distance = 2.3827
  3. Classes 1 ↔ 4: distance = 2.3976

⚖️ Class Imbalance:
  Imbalance ratio: 1.0000
  Samples per class: 20 - 20 (mean: 20.0)

📈 Statistical Confidence:
  Mean accuracy: 82.13%
  95% CI: [78.66%, 85.60%]
  Macro F1: 0.1374
```

## Flags

- `--feature_analysis 1` (default): Enable comprehensive feature analysis
- `--feature_analysis 0`: Disable feature analysis (faster evaluation)
- `--comprehensive_eval 1` (default): Use comprehensive evaluation
- `--comprehensive_eval 0`: Use minimal evaluation

## Ablation Studies

See [ABLATION_STUDIES.md](ABLATION_STUDIES.md) for detailed ablation study configurations.

## Requirements

- numpy
- scipy
- scikit-learn
- torch
- psutil
- GPUtil

All required packages are in `requirements.txt`.
