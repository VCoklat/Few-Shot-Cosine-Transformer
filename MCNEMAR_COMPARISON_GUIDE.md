# McNemar's Test Model Comparison Guide

This guide explains how to use the `compare_models_mcnemar.py` script to statistically compare two model branches or configurations using McNemar's test.

## What is McNemar's Test?

McNemar's test is a statistical test used to determine if there is a statistically significant difference between the error rates of two classification algorithms on the same test set. It focuses on the disagreements between the two classifiers (i.e., instances where one classifier is correct and the other is wrong).

**Key Features:**
- Tests the null hypothesis that both models have the same error rate
- More powerful than simple accuracy comparison for paired samples
- Accounts for the correlation between predictions on the same test set
- Provides p-values to assess statistical significance

## Script Overview

The `compare_models_mcnemar.py` script:
1. Loads two models with different configurations
2. Evaluates both models on the same test set
3. Performs McNemar's test to compare their predictions
4. Generates a detailed comparison report
5. Saves results to JSON for further analysis

## Basic Usage

### Comparing Two Methods with Auto-Detected Checkpoints

```bash
python compare_models_mcnemar.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --method_a FSCT_cosine \
    --method_b FSCT_softmax \
    --checkpoint_dir_a ./checkpoints/miniImagenet/Conv4_FSCT_cosine_5way_1shot \
    --checkpoint_dir_b ./checkpoints/miniImagenet/Conv4_FSCT_softmax_5way_1shot \
    --n_way 5 \
    --k_shot 1 \
    --test_iter 600
```

### Comparing Specific Checkpoint Files

```bash
python compare_models_mcnemar.py \
    --dataset miniImagenet \
    --method_a FSCT_cosine \
    --method_b CTX_softmax \
    --checkpoint_a ./models/branch_t/best_model.tar \
    --checkpoint_b ./models/branch_b/best_model.tar \
    --name_a "Branch T (Transformer)" \
    --name_b "Branch B (Baseline)" \
    --n_way 5 \
    --k_shot 1
```

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--dataset` | Dataset name | `miniImagenet`, `CUB`, `HAM10000` |
| `--method_a` | Method for model A | `FSCT_cosine`, `CTX_softmax` |
| `--method_b` | Method for model B | `FSCT_softmax`, `CTX_cosine` |

You must also provide checkpoints using one of these options for each model:
- `--checkpoint_a` / `--checkpoint_b`: Direct path to checkpoint file
- `--checkpoint_dir_a` / `--checkpoint_dir_b`: Directory containing `best_model.tar`

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backbone` | `Conv4` | Backbone architecture |
| `--n_way` | `5` | Number of classes per episode |
| `--k_shot` | `1` | Number of support samples per class |
| `--n_query` | `16` | Number of query samples per class |
| `--test_iter` | `600` | Number of test episodes |
| `--split` | `novel` | Dataset split (base/val/novel) |
| `--name_a` | `method_a` | Display name for model A |
| `--name_b` | `method_b` | Display name for model B |
| `--output` | `./record/mcnemar_comparison.json` | Output file path |
| `--feti` | `0` | Use FETI pretrained backbone |

## Example Use Cases

### 1. Comparing Branch B and Branch T

If you have two branches/models referred to as "Branch B" (baseline) and "Branch T" (transformer):

```bash
python compare_models_mcnemar.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --method_a CTX_softmax \
    --method_b FSCT_cosine \
    --checkpoint_a ./checkpoints/branch_b.tar \
    --checkpoint_b ./checkpoints/branch_t.tar \
    --name_a "Branch B (Baseline)" \
    --name_b "Branch T (Transformer)" \
    --n_way 5 \
    --k_shot 1 \
    --n_query 16 \
    --test_iter 600 \
    --output ./results/branch_comparison.json
```

### 2. Comparing Cosine vs Softmax Attention

```bash
python compare_models_mcnemar.py \
    --dataset CUB \
    --backbone ResNet18 \
    --method_a FSCT_cosine \
    --method_b FSCT_softmax \
    --checkpoint_dir_a ./checkpoints/CUB/ResNet18_FSCT_cosine_5way_5shot \
    --checkpoint_dir_b ./checkpoints/CUB/ResNet18_FSCT_softmax_5way_5shot \
    --name_a "Cosine Attention" \
    --name_b "Softmax Attention" \
    --n_way 5 \
    --k_shot 5
```

### 3. Medical Dataset Comparison (HAM10000)

```bash
python compare_models_mcnemar.py \
    --dataset HAM10000 \
    --backbone Conv4 \
    --method_a FSCT_cosine \
    --method_b CTX_cosine \
    --checkpoint_a ./medical_models/fsct_ham10000.tar \
    --checkpoint_b ./medical_models/ctx_ham10000.tar \
    --name_a "FSCT on HAM10000" \
    --name_b "CTX on HAM10000" \
    --n_way 7 \
    --k_shot 5 \
    --test_iter 600
```

### 4. Cross-Domain Evaluation

```bash
python compare_models_mcnemar.py \
    --dataset cross \
    --split val \
    --method_a FSCT_cosine \
    --method_b FSCT_cosine \
    --checkpoint_a ./models/trained_on_miniimagenet.tar \
    --checkpoint_b ./models/trained_on_cub.tar \
    --name_a "Trained on miniImagenet" \
    --name_b "Trained on CUB" \
    --n_way 5 \
    --k_shot 1
```

## Output Format

### Console Output

The script provides detailed console output including:

```
================================================================================
McNEMAR'S TEST MODEL COMPARISON
================================================================================
Dataset: miniImagenet
Backbone: Conv4
Task: 5-way 1-shot
Test episodes: 600

Model A: Branch T (Transformer) (FSCT_cosine)
  Checkpoint: ./checkpoints/branch_t.tar

Model B: Branch B (Baseline) (CTX_softmax)
  Checkpoint: ./checkpoints/branch_b.tar
================================================================================

...

================================================================================
McNEMAR'S TEST COMPARISON
================================================================================

Comparing: Branch T (Transformer) vs Branch B (Baseline)

Branch T (Transformer) Accuracy: 0.6543
Branch B (Baseline) Accuracy: 0.6123

Contingency Table:
  Both correct: 5234
  Both wrong: 2543
  Branch T (Transformer) correct, Branch B (Baseline) wrong: 876
  Branch T (Transformer) wrong, Branch B (Baseline) correct: 347

Discordant pairs: 1223
Test type: chi_squared_corrected
Test statistic: 228.4521
P-value: 0.000000

Result: Algorithm A significantly outperforms B (highly significant (p < 0.01))

✓ Branch T (Transformer) performs significantly better than Branch B (Baseline)
================================================================================

...

SUMMARY
================================================================================
Branch T (Transformer): 65.43%
Branch B (Baseline): 61.23%
Accuracy difference: 4.20%

✓ Branch T (Transformer) performs SIGNIFICANTLY BETTER than Branch B (Baseline)
  (p-value: 0.000000 < 0.05)
================================================================================
```

### JSON Output

The script saves detailed results to a JSON file:

```json
{
  "configuration": {
    "dataset": "miniImagenet",
    "backbone": "Conv4",
    "n_way": 5,
    "k_shot": 1,
    "n_query": 16,
    "test_episodes": 600,
    "split": "novel"
  },
  "model_a": {
    "name": "Branch T (Transformer)",
    "method": "FSCT_cosine",
    "checkpoint": "./checkpoints/branch_t.tar",
    "results": {
      "accuracy": 0.6543,
      "confidence_interval_95": {
        "mean": 0.6543,
        "lower": 0.6321,
        "upper": 0.6765,
        "margin": 0.0222
      },
      "kappa": 0.5678,
      "mcc": 0.5712
    }
  },
  "model_b": {
    "name": "Branch B (Baseline)",
    "method": "CTX_softmax",
    "checkpoint": "./checkpoints/branch_b.tar",
    "results": {
      "accuracy": 0.6123,
      "confidence_interval_95": {
        "mean": 0.6123,
        "lower": 0.5901,
        "upper": 0.6345,
        "margin": 0.0222
      },
      "kappa": 0.5154,
      "mcc": 0.5189
    }
  },
  "mcnemar_test": {
    "contingency_table": [2543, 347, 876, 5234],
    "statistic": 228.4521,
    "p_value": 0.0,
    "significant_at_0.05": true,
    "significant_at_0.01": true,
    "algorithm_a_better": true,
    "algorithm_b_better": false,
    "discordant_pairs": 1223,
    "effect_description": "Algorithm A significantly outperforms B (highly significant (p < 0.01))",
    "test_type": "chi_squared_corrected"
  }
}
```

## Interpreting Results

### Understanding the Contingency Table

The contingency table shows how the two models agree/disagree:

| | Model B Correct | Model B Wrong |
|---|---|---|
| **Model A Correct** | Both correct (n11) | A right, B wrong (n10) |
| **Model A Wrong** | A wrong, B right (n01) | Both wrong (n00) |

- **Discordant pairs**: n01 + n10 (instances where models disagree)
- McNemar's test focuses on these disagreements

### Statistical Significance

- **p < 0.01**: Highly significant difference (strong evidence)
- **p < 0.05**: Significant difference (moderate evidence)
- **p >= 0.05**: No significant difference (insufficient evidence)

### Test Types

- **chi_squared_corrected**: Uses Edwards' continuity correction (default)
- **chi_squared**: Chi-squared approximation without correction
- **exact_binomial**: Exact binomial test (used when discordant pairs < 25)

### Effect Description

The script provides a human-readable interpretation:
- "Algorithm A significantly outperforms B"
- "Algorithm B significantly outperforms A"
- "No significant difference between algorithms"

## Tips and Best Practices

1. **Use enough test episodes**: At least 600 episodes for reliable results
2. **Same test set**: Both models must be evaluated on the SAME test set
3. **Check assumptions**: McNemar's test assumes paired samples (same test instances)
4. **Consider practical significance**: A statistically significant difference may not always be practically meaningful
5. **Confidence intervals**: Check if confidence intervals overlap - this provides additional evidence
6. **Multiple comparisons**: If comparing more than 2 models, consider correction methods (e.g., Bonferroni)

## Common Issues

### Issue: Models have different n_way

**Problem**: Models were trained with different n_way values.

**Solution**: Ensure both models use the same n_way for fair comparison, or retrain one model.

### Issue: Checkpoint not found

**Problem**: Checkpoint path is incorrect or file doesn't exist.

**Solution**: 
- Check that paths are correct
- Use `--checkpoint_dir_a/b` to auto-detect `best_model.tar`
- Verify checkpoint structure matches expected format

### Issue: CUDA out of memory

**Problem**: Test episodes too large for GPU memory.

**Solution**:
- Reduce `--test_iter` (minimum 600 recommended)
- Reduce `--n_query`
- Use smaller backbone

### Issue: Inconsistent results

**Problem**: Different runs produce different p-values.

**Solution**:
- Increase `--test_iter` (1000+ for very stable results)
- Set random seed for reproducibility
- Ensure test set is fixed across runs

## Related Scripts

- `ablation_study.py`: Full ablation study framework with multiple configurations
- `eval_utils.py`: Evaluation utilities including McNemar's test functions
- `test.py`: Standard model evaluation script
- `evaluate_f1_scores.py`: Detailed per-class F1 score evaluation

## References

1. McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages". Psychometrika, 12(2), 153-157.

2. Edwards, A. L. (1948). "Note on the correction for continuity in testing the significance of the difference between correlated proportions". Psychometrika, 13(3), 185-187.

3. Dietterich, T. G. (1998). "Approximate statistical tests for comparing supervised classification learning algorithms". Neural Computation, 10(7), 1895-1923.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example use cases
3. Consult the main repository documentation
4. Open an issue on GitHub with details about your configuration
