# Ablation Studies Guide

This guide explains how to perform ablation studies to analyze the contribution of each component in the Few-Shot Cosine Transformer model.

## Overview

Ablation studies help understand which components contribute most to model performance. This repository now includes a dedicated `ablation_study.py` module with:

1. **Ablation Study Framework** - Systematic component ablation and analysis
2. **McNemar's Test** - Statistical comparison of classification algorithms
3. **Automated Reporting** - Generate comprehensive ablation reports

## Quick Start with the Ablation Study Module

### Using the Python API

```python
from ablation_study import (
    AblationStudy,
    AblationConfig,
    AblationType,
    mcnemar_test,
    mcnemar_test_multiple
)

# Create an ablation study
study = AblationStudy(baseline_name="Full Model")

# Add results (after evaluating each configuration)
study.add_result(
    name="Full Model",
    config=AblationConfig(
        name="Full Model",
        ablation_type=AblationType.FULL_MODEL,
        description="Complete model with all components",
        method="FSCT_cosine"
    ),
    accuracy=0.7342,
    std=0.015,
    predictions=preds_full,
    true_labels=labels,
    is_baseline=True
)

study.add_result(
    name="Without Cosine Attention",
    config=AblationConfig(
        name="Without Cosine Attention",
        ablation_type=AblationType.NO_COSINE_ATTENTION,
        description="Using softmax instead of cosine attention",
        method="FSCT_softmax"
    ),
    accuracy=0.7023,
    std=0.018,
    predictions=preds_softmax,
    true_labels=labels
)

# Generate report
print(study.generate_report())

# Save results
study.save_results("ablation_results.json")
```

### Command-Line Usage

The following CLI arguments are available for ablation study configurations:

#### Basic Ablation Study Arguments

```bash
# Run ablation study comparing methods
python test.py --method FSCT_cosine --dataset miniImagenet --ablation_study 1 \
    --mcnemar_compare "FSCT_cosine,FSCT_softmax,CTX_cosine" \
    --ablation_output "./record/ablation_results.json"
```

#### Ablation Configuration Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_heads` | int | 8 | Number of attention heads for transformer models |
| `--use_se` | int | 1 | Enable (1) or disable (0) SE (Squeeze-and-Excitation) blocks |
| `--use_vic` | int | 1 | Enable (1) or disable (0) VIC regularization |
| `--use_dynamic_weights` | int | 1 | Enable (1) or disable (0) dynamic prototype weighting |
| `--vic_components` | str | "all" | VIC components: "all", "variance", "invariance", "covariance", or comma-separated |

#### Examples

```bash
# Model with single attention head
python train.py --method FSCT_cosine --dataset miniImagenet --num_heads 1

# Model with 4 attention heads
python train.py --method FSCT_cosine --dataset miniImagenet --num_heads 4

# Model without dynamic weighting
python train.py --method FSCT_cosine --dataset miniImagenet --use_dynamic_weights 0

# Model without SE blocks (placeholder for future implementation)
python train.py --method FSCT_cosine --dataset miniImagenet --use_se 0

# Model without VIC regularization (placeholder for future implementation)
python train.py --method FSCT_cosine --dataset miniImagenet --use_vic 0

# Model with only variance component of VIC
python train.py --method FSCT_cosine --dataset miniImagenet --vic_components variance

# Model with variance and covariance VIC components
python train.py --method FSCT_cosine --dataset miniImagenet --vic_components "variance,covariance"

# Combined ablation: 4 heads without dynamic weights
python train.py --method FSCT_cosine --dataset miniImagenet --num_heads 4 --use_dynamic_weights 0
```

## McNemar's Test

McNemar's test is used for comparing two classification algorithms, focusing on the differences in their error rates. It specifically examines instances where one algorithm got right what the other got wrong.

### When to Use McNemar's Test

- Comparing two models on the **same test set**
- Determining if performance differences are **statistically significant**
- Ablation studies where you want to validate component importance

### How It Works

The test constructs a 2x2 contingency table:

|                  | Model B Correct | Model B Wrong |
|------------------|-----------------|---------------|
| Model A Correct  | n11             | n10           |
| Model A Wrong    | n01             | n00           |

- **n01**: Instances A got wrong but B got right
- **n10**: Instances A got right but B got wrong
- **Discordant pairs**: n01 + n10 (cases where models disagree)

The null hypothesis is that both models have the same error rate.

### Using McNemar's Test

```python
from ablation_study import mcnemar_test, format_contingency_table

# Compare two sets of predictions
result = mcnemar_test(
    predictions_a=model_a_preds,  # Predictions from model A
    predictions_b=model_b_preds,  # Predictions from model B
    true_labels=true_labels,       # Ground truth
    correction=True                # Apply continuity correction
)

# Results include:
print(f"P-value: {result['p_value']:.6f}")
print(f"Significant at 0.05: {result['significant_at_0.05']}")
print(f"Effect: {result['effect_description']}")

# Format contingency table
n00, n01, n10, n11 = result['contingency_table']
print(format_contingency_table(n00, n01, n10, n11, "FSCT_cosine", "FSCT_softmax"))
```

### Multiple Model Comparison

```python
from ablation_study import mcnemar_test_multiple

# Compare multiple models pairwise
results = mcnemar_test_multiple(
    predictions_list=[preds_a, preds_b, preds_c],
    algorithm_names=["FSCT_cosine", "FSCT_softmax", "CTX_cosine"],
    true_labels=true_labels
)

# Access pairwise comparisons
for comparison in results['pairwise_comparisons']:
    print(f"{comparison['algorithm_a']} vs {comparison['algorithm_b']}: "
          f"p={comparison['p_value']:.4f}")
```

### Integration with Evaluation

```python
import eval_utils

# Evaluate models and get predictions
res_a, preds_a, labels = eval_utils.evaluate_with_predictions(
    test_loader, model_a, n_way, device=device
)
res_b, preds_b, _ = eval_utils.evaluate_with_predictions(
    test_loader, model_b, n_way, device=device
)

# Compare using McNemar's test
comparison = eval_utils.compare_models_mcnemar(
    preds_a, preds_b, labels,
    model_a_name="FSCT_cosine",
    model_b_name="FSCT_softmax"
)

# Print formatted comparison
eval_utils.print_mcnemar_comparison(comparison)
```

## Ablation Study Configurations

### 1. Model without SE Blocks (Channel Attention)

To measure the contribution of Squeeze-and-Excitation (SE) blocks for channel attention:

**Configuration:**
- Disable SE blocks in the backbone architecture
- Keep all other components active

**Command:**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --use_se 0
python test.py --method FSCT_cosine --dataset miniImagenet --use_se 0
```

> Note: SE block support is a placeholder for future implementation in the backbone.

### 2. Model without Cosine Attention

To measure the impact of cosine attention vs. standard dot-product attention:

**Configuration:**
- Use `FSCT_softmax` or `CTX_softmax` instead of `FSCT_cosine` or `CTX_cosine`

**Command:**
```bash
python train.py --method FSCT_softmax --dataset miniImagenet --n_way 5 --k_shot 5
python test.py --method FSCT_softmax --dataset miniImagenet --n_way 5 --k_shot 5
```

### 3. Model without VIC Regularization

To measure the contribution of Variance-Invariance-Covariance regularization:

**Configuration:**
- Disable VIC loss terms during training

**Command:**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --use_vic 0
python test.py --method FSCT_cosine --dataset miniImagenet --use_vic 0
```

> Note: VIC regularization support is a placeholder for future implementation.

### 4. Model without Dynamic Weighting

To evaluate the impact of dynamic weight adjustment:

**Configuration:**
- Use fixed uniform weights instead of learned/adaptive weights
- The model uses simple mean prototypes instead of learned weighted prototypes

**Command:**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --use_dynamic_weights 0
python test.py --method FSCT_cosine --dataset miniImagenet --use_dynamic_weights 0
```

### 5. Model with Single VIC Component

To analyze individual VIC components:

**Variance Only:**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --vic_components variance
```

**Covariance Only:**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --vic_components covariance
```

**Invariance Only:**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --vic_components invariance
```

**Variance + Covariance:**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --vic_components "variance,covariance"
```

> Note: VIC component selection is a placeholder for future implementation.

### 6. Varying Number of Attention Heads

To find the optimal number of attention heads:

**Single Head (1):**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --num_heads 1
python test.py --method FSCT_cosine --dataset miniImagenet --num_heads 1
```

**Two Heads:**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --num_heads 2
python test.py --method FSCT_cosine --dataset miniImagenet --num_heads 2
```

**Four Heads (recommended optimal):**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --num_heads 4
python test.py --method FSCT_cosine --dataset miniImagenet --num_heads 4
```

**Eight Heads (default):**
```bash
python train.py --method FSCT_cosine --dataset miniImagenet --num_heads 8
python test.py --method FSCT_cosine --dataset miniImagenet --num_heads 8
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
   - Compare accuracy, confusion matrix, and other metrics
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
3. **Per-Class Analysis:** Use confusion matrix and per-class precision/recall for per-class performance
4. **Confusion Matrix:** Error patterns
5. **Feature Quality Metrics:**
   - Feature collapse rate
   - Feature utilization
   - Redundancy levels
   - Intra-class consistency
6. **McNemar's Test Results:**
   - P-value for statistical significance
   - Discordant pairs count
   - Effect direction (which model is better)

### Performance Drop Analysis

Calculate the performance drop for each ablation:

```
Performance Drop = (Baseline Accuracy - Ablation Accuracy) / Baseline Accuracy * 100%
```

This indicates the contribution of each component.

### Example Results Table

| Configuration | Accuracy | Δ from Baseline | p-value (McNemar) | Significant |
|--------------|----------|-----------------|-------------------|-------------|
| Full Model (Baseline) | 73.42% | - | - | - |
| w/o SE Blocks | 71.85% | -1.57% | 0.023 | ✓ |
| w/o Cosine Attention | 70.23% | -3.19% | 0.001 | ✓✓ |
| w/o VIC Regularization | 72.10% | -1.32% | 0.087 | - |
| w/o Dynamic Weighting | 71.50% | -1.92% | 0.015 | ✓ |
| 1 Head | 69.80% | -3.62% | <0.001 | ✓✓ |
| 2 Heads | 71.20% | -2.22% | 0.004 | ✓✓ |
| 8 Heads | 72.90% | -0.52% | 0.312 | - |

✓ = p < 0.05, ✓✓ = p < 0.01

## Advanced Analysis

### Feature Space Analysis

When running ablation studies with `--feature_analysis 1`, pay attention to:

1. **Feature Collapse:** Does removing a component cause more feature collapse?
2. **Redundancy:** Does the ablation increase feature redundancy?
3. **Consistency:** How does intra-class consistency change?
4. **Class Separability:** Check confusing class pairs

### Statistical Significance with McNemar's Test

The ablation study module now provides McNemar's test for rigorous statistical comparison:

```python
from ablation_study import AblationStudy

# After adding results to the study
impacts = study.compute_ablation_impact()

for impact in impacts:
    print(f"{impact['name']}:")
    print(f"  Accuracy drop: {impact['absolute_difference']:.4f}")
    print(f"  Significant: {impact['is_significant']}")
    print(f"  p-value: {impact['p_value']:.6f}")
```

### Interpreting McNemar's Test Results

- **p < 0.05**: Statistically significant difference
- **p < 0.01**: Highly significant difference  
- **p ≥ 0.05**: No significant difference (could be due to chance)

When **discordant pairs < 25**, the exact binomial test is used instead of chi-squared approximation for more accurate results.

## Tips for Ablation Studies

1. **Keep Everything Else Constant:** Only change one component at a time
2. **Document Changes:** Keep detailed notes of modifications
3. **Use Version Control:** Git branch for each ablation
4. **Monitor Training:** Check if training dynamics change
5. **Resource Planning:** Some ablations may require different computational resources
6. **Use McNemar's Test:** Validate that differences are statistically significant

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

### Using the Ablation Study Module

```python
from ablation_study import AblationStudy

# Create study and add results
study = AblationStudy(baseline_name="Full Model")

# ... add results ...

# Generate comprehensive report
print(study.generate_report())

# Save to JSON for later analysis
study.save_results("ablation_results.json")
```

## API Reference

### ablation_study.py

#### Classes

- `AblationType` - Enum of ablation types (FULL_MODEL, NO_COSINE_ATTENTION, etc.)
- `AblationConfig` - Configuration dataclass for ablation experiments
- `AblationResult` - Result dataclass for ablation experiments
- `AblationStudy` - Main class for managing ablation studies

#### Functions

- `mcnemar_test(predictions_a, predictions_b, true_labels, correction=True, exact=False)` - Perform McNemar's test
- `mcnemar_test_multiple(predictions_list, algorithm_names, true_labels)` - Pairwise McNemar's tests
- `compute_contingency_table(predictions_a, predictions_b, true_labels)` - Compute contingency table
- `format_contingency_table(n00, n01, n10, n11)` - Format table as readable string
- `compute_performance_drop(baseline_accuracy, ablation_accuracy)` - Calculate ablation impact
- `rank_component_importance(ablation_results)` - Rank components by importance

### eval_utils.py Extensions

- `evaluate_with_predictions()` - Evaluate and return predictions for McNemar's test
- `compare_models_mcnemar()` - Compare two models using McNemar's test
- `print_mcnemar_comparison()` - Print formatted comparison result
- `run_ablation_comparison()` - Run full ablation comparison across multiple models

## Conclusion

Ablation studies provide crucial insights into model design choices. By systematically removing or modifying components, you can:

- Identify critical components
- Guide future improvements
- Justify design decisions
- Optimize model complexity
- **Validate findings with statistical significance tests**

Remember to run comprehensive evaluations for each ablation to get complete understanding of component contributions, and use McNemar's test to ensure observed differences are statistically significant.
