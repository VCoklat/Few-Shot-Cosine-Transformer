# Comprehensive Evaluation Metrics

This document describes the comprehensive evaluation metrics and feature analysis capabilities added to the Few-Shot Cosine Transformer repository.

## Overview

The evaluation system has been significantly extended to provide deep insights into model performance, feature quality, and potential improvements. These metrics are essential for:

- Understanding model behavior beyond simple accuracy
- Identifying potential issues in feature representations
- Guiding model architecture decisions
- Conducting thorough ablation studies
- Publishing comprehensive experimental results

## Features

### 1. Core Evaluation Metrics

#### 95% Confidence Interval
- **Purpose**: Provides statistical confidence in performance estimates
- **Implementation**: Computed from per-episode accuracies using normal approximation
- **Output**: Mean accuracy with lower and upper bounds at 95% confidence
- **Use case**: Essential for comparing models and establishing statistical significance

#### Per-Class F1 Score
- **Purpose**: Harmonic mean of precision and recall for each class
- **Implementation**: Uses sklearn's f1_score with per-class averaging
- **Output**: F1 score for each class plus macro-averaged F1
- **Use case**: Identifying which classes are well-learned vs. problematic

#### Enhanced Confusion Matrix
- **Purpose**: Detailed analysis of classification errors
- **Implementation**: Standard confusion matrix with per-class accuracy breakdown
- **Output**: Matrix visualization plus per-class statistics
- **Use case**: Understanding error patterns and class confusions

### 2. Feature Space Analysis

#### Feature Collapse Detection
- **Purpose**: Identify dimensions with minimal variance (< 1e-4 standard deviation)
- **Method**: Compute standard deviation per dimension, flag those below threshold
- **Metrics**:
  - Number and ratio of collapsed dimensions
  - Min/max/mean standard deviation across dimensions
- **Interpretation**: High collapse indicates wasted representational capacity

#### Feature Utilization
- **Purpose**: Measure how effectively each feature dimension is being used
- **Method**: Compare actual value distribution to maximum possible range
- **Metrics**:
  - Mean utilization score per dimension
  - Number of low-utilization dimensions (< 30%)
  - Utilization statistics (min/max/std)
- **Interpretation**: Low utilization suggests underutilized feature capacity

#### Diversity Score
- **Purpose**: Quantify within-class variation using coefficient of variation
- **Method**: Compute CV of distances from samples to class centroids
- **Metrics**:
  - Mean diversity score across classes
  - Per-class diversity scores
- **Interpretation**: Balanced diversity indicates healthy feature distributions

### 3. Feature Redundancy Analysis

#### Correlation Analysis
- **Purpose**: Detect highly correlated (redundant) feature pairs
- **Method**: Compute Pearson correlation between all feature pairs
- **Metrics**:
  - Number of high correlation pairs (> 0.9)
  - Number of moderate correlation pairs (> 0.7)
  - Mean absolute correlation
- **Interpretation**: High redundancy suggests over-parameterization

#### Effective Dimensionality (PCA)
- **Purpose**: Determine how many dimensions capture 95% of variance
- **Method**: Apply PCA and find cumulative variance threshold
- **Metrics**:
  - Effective dimensions at 95% variance
  - Dimensionality reduction ratio
- **Interpretation**: Large reduction possible indicates redundancy

### 4. Class Relationship Analysis

#### Intra-Class Consistency
- **Purpose**: Measure uniformity of representations within each class
- **Method**: Combine Euclidean distances and cosine similarities within classes
- **Metrics**:
  - Mean Euclidean consistency
  - Mean cosine consistency
  - Combined consistency score
  - Per-class consistency values
- **Interpretation**: High consistency = tight, well-formed class clusters

#### Confusing Class Pairs
- **Purpose**: Identify classes that are hard to distinguish
- **Method**: Compute inter-centroid distances between all class pairs
- **Metrics**:
  - Top-k most confusing pairs (smallest distances)
  - Mean/std/min/max inter-centroid distances
- **Interpretation**: Small distances indicate difficult class separations

#### Class Imbalance Ratio
- **Purpose**: Quantify sample distribution across classes
- **Method**: Ratio of minority to majority class sample counts
- **Metrics**:
  - Imbalance ratio
  - Min/max/mean class sample counts
  - Per-class sample counts
- **Interpretation**: Low ratio indicates significant imbalance issues

## Usage

### Basic Comprehensive Evaluation

Run standard evaluation with confidence intervals, F1 scores, and confusion matrix:

```bash
python test.py --dataset miniImagenet \
               --method FSCT_cosine \
               --n_way 5 \
               --k_shot 5 \
               --comprehensive_eval 1
```

### With Feature Analysis

Add complete feature space analysis:

```bash
python test.py --dataset miniImagenet \
               --method FSCT_cosine \
               --n_way 5 \
               --k_shot 5 \
               --comprehensive_eval 1 \
               --feature_analysis 1
```

### Programmatic Usage

```python
import eval_utils

# Standard comprehensive evaluation
results = eval_utils.evaluate(test_loader, model, n_way=5, 
                             class_names=['Class_0', 'Class_1', ...])

# Display results
eval_utils.pretty_print(results)

# With feature analysis
results = eval_utils.evaluate_comprehensive(test_loader, model, n_way=5)
eval_utils.pretty_print(results, show_feature_analysis=True)
```

### Using Individual Metrics

```python
from feature_analysis import (
    compute_confidence_interval,
    detect_feature_collapse,
    compute_feature_utilization,
    comprehensive_feature_analysis
)

# Example: Confidence interval
accuracies = np.array([0.72, 0.73, 0.71, ...])  # Per-episode accuracies
mean, lower, upper = compute_confidence_interval(accuracies)
print(f"Accuracy: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

# Example: Feature collapse
features = model.extract_features(data)  # Shape: (n_samples, n_features)
collapse_info = detect_feature_collapse(features)
print(f"Collapsed: {collapse_info['collapsed_dimensions']} dims")

# Example: Full analysis
labels = np.array([0, 0, 1, 1, 2, 2, ...])  # Class labels
analysis = comprehensive_feature_analysis(features, labels)
```

## Output Format

### Standard Output Example

```
================================================================================
CLASSIFICATION METRICS
================================================================================

Accuracy: 0.7342 (73.42%)
95% Confidence Interval: [0.7275, 0.7409]
  (±0.0067 or ±0.67%)
  Based on 600 episodes

Macro-F1: 0.7340
Macro Precision: 0.7350
Macro Recall: 0.7330

Per-Class F1 Scores:
  Way 0: 0.7420
  Way 1: 0.7280
  Way 2: 0.7340
  Way 3: 0.7310
  Way 4: 0.7350

Cohen's κ: 0.6678
Matthews CorrCoef: 0.6685
Top-5 Accuracy: 0.9523

================================================================================
CONFUSION MATRIX
================================================================================
[[147   8   5   3   2]
 [  9 142   6   5   3]
 [  6   7 145   4   3]
 [  4   6   5 143   7]
 [  3   4   4   8 146]]

================================================================================
FEATURE SPACE ANALYSIS
================================================================================

Feature Collapse Detection:
  Collapsed dimensions: 12/512 (2.3%)
  Std range: [0.000023, 2.145678], mean: 0.987654

Feature Utilization:
  Mean utilization: 0.7845
  Low utilization dims (<30%): 34

Diversity Score:
  Mean diversity (CV): 0.4521
  Std diversity: 0.0876

Feature Redundancy:
  Total features: 512
  Effective dimensions (95% variance): 287
  Dimensionality reduction ratio: 0.5605
  High correlation pairs (>0.9): 15
  Moderate correlation pairs (>0.7): 47

Intra-Class Consistency:
  Mean Euclidean consistency: 0.7234
  Mean Cosine consistency: 0.8123
  Mean Combined consistency: 0.7679

Most Confusing Class Pairs (closest centroids):
  Way 0 ↔ Way 3: distance = 1.2345
  Way 1 ↔ Way 4: distance = 1.3456
  Way 2 ↔ Way 3: distance = 1.4567

Class Imbalance:
  Imbalance ratio: 0.9818
  Min class samples: 163
  Max class samples: 166
```

## Integration with Ablation Studies

See [ABLATION_STUDIES.md](ABLATION_STUDIES.md) for detailed guidance on using these metrics for ablation studies.

Key workflow:
1. Run baseline with full feature analysis
2. Run each ablation configuration
3. Compare metrics to identify critical components
4. Use feature analysis to understand why components matter

## Examples

Run the example script to see all metrics in action:

```bash
python example_comprehensive_metrics.py
```

This demonstrates:
- Confidence interval calculation
- Feature collapse detection
- Redundancy analysis
- Consistency metrics
- Confusing pair identification
- Full comprehensive analysis

## Technical Details

### Implementation Files

- **feature_analysis.py**: Core analysis functions for feature space metrics
- **eval_utils.py**: Extended evaluation with confidence intervals and integration
- **test.py**: Updated test script with flags for comprehensive evaluation
- **io_utils.py**: Added command-line arguments for feature analysis flags

### Dependencies

- numpy: Numerical operations
- scipy: Statistical functions and distance metrics
- scikit-learn: PCA, metrics, and machine learning utilities
- torch: Model evaluation (already required)

No additional dependencies beyond the existing requirements.txt

### Performance Considerations

- Feature extraction may require additional memory for large datasets
- Comprehensive analysis adds ~10-20% to evaluation time
- Use `--feature_analysis 0` to disable when not needed
- Feature extraction can be done in chunks to manage memory

## Validation

Run the test suite to verify installation:

```bash
python test_comprehensive_eval.py
```

This validates:
- All metric functions work correctly
- Feature analysis module is properly imported
- Integration with eval_utils is functional
- Example data produces expected outputs

## Citation

If you use these comprehensive evaluation metrics in your research, please cite:

```bibtex
@article{nguyen2023FSCT,
  author={Nguyen, Quang-Huy and Nguyen, Cuong Q. and Le, Dung D. and Pham, Hieu H.},
  journal={IEEE Access}, 
  title={Enhancing Few-Shot Image Classification With Cosine Transformer}, 
  year={2023},
  volume={11},
  pages={79659-79672},
  doi={10.1109/ACCESS.2023.3298299}
}
```

## Support

For questions or issues related to the comprehensive evaluation metrics:
1. Check this documentation
2. Review example_comprehensive_metrics.py
3. See ABLATION_STUDIES.md for ablation study guidance
4. Open an issue on GitHub with the "evaluation" label
