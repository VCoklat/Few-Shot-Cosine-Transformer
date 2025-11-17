# Statistical Significance Testing and Per-Class F1 Score Analysis

This document describes the new statistical significance testing and per-class F1 score comparison features added to the Few-Shot Cosine Transformer repository.

## Overview

The repository now includes comprehensive tools for:
1. **Per-Class F1 Score Analysis** - Detailed performance metrics for each class
2. **Statistical Significance Testing** - Rigorous statistical comparison between models
3. **Multiple Test Types** - McNemar's test, paired t-test, Wilcoxon test

## Features

### 1. Per-Class F1 Score Computation

Calculate F1 scores for each class to identify which classes are well-classified and which need improvement.

**Key Metrics:**
- Per-class F1, Precision, and Recall
- Macro F1 (unweighted average across classes)
- Micro F1 (equivalent to accuracy)
- Weighted F1 (weighted by class support)
- Standard deviation and range of F1 scores

### 2. Statistical Significance Tests

#### McNemar's Test
- **Purpose:** Compare paired predictions from two models
- **Use case:** Determine if one model is significantly better than another
- **Output:** Chi-square statistic, p-value, contingency table
- **Interpretation:** Tests whether the disagreements between models are significant

#### Paired t-test
- **Purpose:** Compare episode-wise accuracies
- **Use case:** Test if mean accuracy difference is significant
- **Output:** t-statistic, p-value, confidence interval
- **Interpretation:** Parametric test assuming normal distribution

#### Wilcoxon Signed-Rank Test
- **Purpose:** Non-parametric alternative to paired t-test
- **Use case:** When normality assumption is violated
- **Output:** Test statistic, p-value, median difference
- **Interpretation:** More robust to outliers and non-normal distributions

### 3. Per-Class F1 Comparison
- Compare F1 scores between two models for each class
- Identify which classes improve/degrade when switching models
- Statistical test for overall F1 difference across classes

## Usage

### Basic Per-Class F1 Score Analysis

```python
from significance_test import compute_per_class_f1

# After evaluation, you have y_true and y_pred
f1_results = compute_per_class_f1(y_true, y_pred, n_classes=5)

print(f"Macro F1: {f1_results['macro_f1']:.4f}")
print(f"Micro F1: {f1_results['micro_f1']:.4f}")

for i, f1 in enumerate(f1_results['per_class_f1']):
    print(f"Class {i}: F1={f1:.4f}")
```

### Comparing Two Models

```python
from significance_test import comprehensive_significance_test, pretty_print_significance_test

# Compare two models
results = comprehensive_significance_test(
    y_true, 
    y_pred_model1, 
    y_pred_model2,
    accuracies1,  # Episode-wise accuracies (optional)
    accuracies2,  # Episode-wise accuracies (optional)
    class_names=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"],
    model1_name="FSCT_cosine",
    model2_name="CTX_cosine"
)

# Print comprehensive results
pretty_print_significance_test(results)
```

### Integration with Existing Evaluation

The per-class F1 scores are automatically computed when using `eval_utils.evaluate()`:

```python
from eval_utils import evaluate, pretty_print

# Standard evaluation
results = evaluate(test_loader, model, n_way=5, class_names=class_names)

# Display results (includes per-class F1 scores)
pretty_print(results)
```

## Files

### Core Implementation
- **`significance_test.py`** - Main implementation of all significance tests
  - `mcnemar_test()` - McNemar's test
  - `paired_ttest()` - Paired t-test
  - `wilcoxon_test()` - Wilcoxon signed-rank test
  - `compute_per_class_f1()` - Per-class F1 computation
  - `compare_per_class_f1()` - F1 comparison between models
  - `comprehensive_significance_test()` - All-in-one testing
  - `pretty_print_significance_test()` - Formatted output

### Testing and Examples
- **`test_significance.py`** - Unit tests for all significance tests
- **`example_significance_test.py`** - Comprehensive examples demonstrating usage

### Integration
- **`eval_utils.py`** - Updated to include per-class F1 scores in evaluation

## Example Output

### Per-Class F1 Scores
```
================================================================================
PER-CLASS F1 SCORES
================================================================================

Macro F1: 0.8262
Micro F1: 0.8267
Weighted F1: 0.8262
Std F1: 0.0463
Range: [0.7586, 0.8966]

F1 Score by Class:
  Way 0: F1=0.8485 (Precision=0.7778, Recall=0.9333)
  Way 1: F1=0.8966 (Precision=0.9286, Recall=0.8667)
  Way 2: F1=0.8276 (Precision=0.8571, Recall=0.8000)
  Way 3: F1=0.8000 (Precision=0.8000, Recall=0.8000)
  Way 4: F1=0.7586 (Precision=0.7857, Recall=0.7333)
```

### Statistical Comparison
```
================================================================================
STATISTICAL SIGNIFICANCE TEST RESULTS
================================================================================

Comparing: Few-Shot Cosine Transformer vs Cross Transformer

--------------------------------------------------------------------------------
McNemar's Test (Paired Predictions)
--------------------------------------------------------------------------------
Statistic: 6.0631
P-value: 0.0138
Significant (α=0.05): YES

Contingency Table:
  Both correct: 527
  Only Few-Shot Cosine Transformer correct: 223
  Only Cross Transformer correct: 173
  Both wrong: 77

Interpretation: Model 1 wins 223 cases, Model 2 wins 173 cases. 
Significant difference (p=0.0138).

--------------------------------------------------------------------------------
Per-Class F1 Score Comparison
--------------------------------------------------------------------------------

Few-Shot Cosine Transformer:
  Macro F1: 0.7500
  Micro F1: 0.7500
  Std F1: 0.0118

Cross Transformer:
  Macro F1: 0.6995
  Micro F1: 0.7000
  Std F1: 0.0261

Comparison:
  Mean F1 difference: 0.0504
  Std F1 difference: 0.0336
  T-statistic: 3.0046
  P-value: 0.0398
  Significant (α=0.05): YES

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
McNemar's test significant: YES
Per-class F1 significant: YES
Overall significant difference: YES

Based on McNemar's test (p=0.0138) and per-class F1 comparison (p=0.0398), 
there is a statistically significant difference between Few-Shot Cosine 
Transformer and Cross Transformer.
================================================================================
```

## Statistical Interpretation

### Understanding P-values
- **p < 0.05**: Statistically significant (conventional threshold)
- **p < 0.01**: Highly significant
- **p < 0.001**: Very highly significant
- **p ≥ 0.05**: Not significant (difference could be due to chance)

### Choosing the Right Test

1. **Use McNemar's Test when:**
   - Comparing predictions from two models on the same test set
   - You want to test if disagreements are significant
   - You have paired nominal data (correct/incorrect)

2. **Use Paired t-test when:**
   - Comparing episode-wise accuracies
   - Data is approximately normally distributed
   - You have continuous measurements (accuracy per episode)

3. **Use Wilcoxon Test when:**
   - Comparing episode-wise accuracies
   - Data is not normally distributed
   - You want a non-parametric alternative to t-test

### Interpreting Results

**Significant difference found (p < 0.05):**
- The performance difference is unlikely due to random chance
- You can confidently say one model is better than the other
- Safe to prefer the better-performing model

**No significant difference (p ≥ 0.05):**
- The performance difference could be due to random chance
- Cannot confidently say one model is better
- Consider other factors: inference time, memory, interpretability
- May need more test data for higher statistical power

## Best Practices

1. **Always report confidence intervals** along with point estimates
2. **Use multiple tests** for robust conclusions (McNemar + t-test)
3. **Consider effect size** not just statistical significance
4. **Check per-class performance** to identify specific strengths/weaknesses
5. **Run sufficient episodes** (≥1000 recommended) for reliable statistics
6. **Account for multiple comparisons** if testing many model pairs

## Dependencies

The significance testing module requires:
- `numpy` - Array operations and basic statistics
- `scipy` - Advanced statistical tests
- `scikit-learn` - Already included for other metrics

All dependencies are already in `requirements.txt`.

## Quick Start

1. **Run the tests:**
   ```bash
   python test_significance.py
   ```

2. **See examples:**
   ```bash
   python example_significance_test.py --example all
   ```

3. **Use in your evaluation:**
   ```python
   from significance_test import compute_per_class_f1
   
   # After getting predictions
   f1_results = compute_per_class_f1(y_true, y_pred, n_classes)
   print(f"Macro F1: {f1_results['macro_f1']:.4f}")
   ```

## References

1. McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages". Psychometrika.
2. Student (1908). "The probable error of a mean". Biometrika.
3. Wilcoxon, F. (1945). "Individual comparisons by ranking methods". Biometrics Bulletin.

## Contact

For questions or issues related to significance testing, please open an issue on GitHub or contact the repository maintainers.
