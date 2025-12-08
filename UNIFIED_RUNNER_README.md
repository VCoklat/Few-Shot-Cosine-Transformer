# Unified Experiment Runner

A comprehensive framework for running experiments with the Few-Shot Cosine Transformer, including training, testing, ablation studies, qualitative analysis, and statistical significance testing.

## Overview

The unified experiment runner (`run_experiments.py`) provides a single interface to:

1. **Train and Test** baseline and proposed models with comprehensive metrics
2. **Qualitative Analysis** with visualizations (t-SNE, confusion matrices)
3. **Ablation Studies** with 8 different VIC component configurations
4. **McNemar's Test** for statistical significance testing
5. **Feature Analysis** including collapse detection and dimensionality analysis

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages:
- torch, torchvision
- numpy, scipy
- matplotlib, seaborn
- scikit-learn
- tqdm
- GPUtil, psutil

## Quick Start

### Run All Experiments

```bash
python run_experiments.py \
  --dataset miniImagenet \
  --backbone Conv4 \
  --n_way 5 \
  --k_shot 1 \
  --run_mode all
```

### Run Specific Modes

**Training and Testing Only:**
```bash
python run_experiments.py \
  --dataset CUB \
  --backbone ResNet18 \
  --n_way 5 \
  --k_shot 5 \
  --run_mode train_test
```

**Ablation Study Only:**
```bash
python run_experiments.py \
  --dataset miniImagenet \
  --backbone Conv4 \
  --run_mode ablation
```

**Qualitative Analysis Only:**
```bash
python run_experiments.py \
  --dataset CIFAR \
  --backbone Conv4 \
  --run_mode qualitative
```

**Feature Collapse Analysis:**
```bash
python run_experiments.py \
  --dataset miniImagenet \
  --backbone Conv4 \
  --run_mode feature_analysis
```

**McNemar's Statistical Testing:**
```bash
python run_experiments.py \
  --dataset miniImagenet \
  --backbone Conv4 \
  --run_mode mcnemar
```

## Command-Line Arguments

### Dataset and Model Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `miniImagenet` | Dataset selection (miniImagenet, CUB, CIFAR, Omniglot, etc.) |
| `--backbone` | `Conv4` | Backbone architecture (Conv4, Conv6, ResNet10, ResNet18, ResNet34) |
| `--n_way` | `5` | Number of classes per episode |
| `--k_shot` | `1` | Number of support samples per class |
| `--n_query` | `16` | Number of query samples per class |

### Training Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_epochs` | `50` | Number of training epochs |
| `--learning_rate` | `1e-3` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay for regularization |
| `--optimization` | `AdamW` | Optimizer (Adam, AdamW, SGD) |

### Testing Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_iter` | `600` | Number of test episodes |

### Run Mode

| Argument | Options | Description |
|----------|---------|-------------|
| `--run_mode` | `all`, `train_test`, `ablation`, `qualitative`, `feature_analysis`, `mcnemar` | Select which experiments to run |

### Output Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `./results` | Directory for saving results |
| `--seed` | `4040` | Random seed for reproducibility |

### Advanced Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--baseline_checkpoint` | `None` | Path to baseline checkpoint (skip training) |
| `--proposed_checkpoint` | `None` | Path to proposed checkpoint (skip training) |
| `--ablation_experiments` | `All (E1-E8)` | Comma-separated list of experiments (e.g., `E1,E2,E6`) |

## Ablation Experiments

The runner supports 8 pre-configured ablation experiments:

| ID | Name | Invariance | Covariance | Variance | Dynamic | Description |
|----|------|------------|------------|----------|---------|-------------|
| E1 | Full | ✓ | ✓ | ✓ | ✓ | Full Dynamic VIC model |
| E2 | InvDyn | ✓ | ✗ | ✗ | ✓ | Only invariance + dynamic |
| E3 | InvCovDyn | ✓ | ✓ | ✗ | ✓ | Invariance + covariance + dynamic |
| E4 | InvVarDyn | ✓ | ✗ | ✓ | ✓ | Invariance + variance + dynamic |
| E5 | FullNoD | ✓ | ✓ | ✓ | ✗ | Full VIC without dynamic |
| E6 | Baseline | ✗ | ✗ | ✗ | ✗ | Baseline cosine similarity |
| E7 | CovDyn | ✗ | ✓ | ✗ | ✓ | Only covariance + dynamic |
| E8 | VarDyn | ✗ | ✗ | ✓ | ✓ | Only variance + dynamic |

### Running Specific Ablation Experiments

```bash
python run_experiments.py \
  --dataset miniImagenet \
  --backbone Conv4 \
  --run_mode ablation \
  --ablation_experiments E1,E2,E6
```

## Output Structure

Results are organized in a structured directory format:

```
results/
└── {dataset}_{backbone}_{n_way}w{k_shot}s/
    ├── experiment_config.json
    ├── quantitative/
    │   ├── baseline_results.json
    │   ├── proposed_results.json
    │   ├── comparison_metrics.json
    │   └── best_model.tar
    ├── qualitative/
    │   ├── tsne_baseline.png
    │   ├── tsne_proposed.png
    │   ├── confusion_matrix_baseline.png
    │   └── confusion_matrix_proposed.png
    ├── ablation/
    │   ├── ablation_results.json
    │   ├── ablation_comparison.png
    │   ├── component_importance.json
    │   └── component_importance.png
    ├── mcnemar/
    │   ├── significance_tests.json
    │   ├── contingency_tables.png
    │   └── pairwise_comparison_matrix.png
    └── feature_analysis/
        ├── feature_collapse_metrics.json
        ├── variance_distribution_baseline.png
        ├── variance_distribution_proposed.png
        ├── correlation_matrix_baseline.png
        └── correlation_matrix_proposed.png
```

## Output Files

### Quantitative Results

**baseline_results.json / proposed_results.json:**
```json
{
  "accuracy": 0.xxxx,
  "std": 0.xxxx,
  "precision": 0.xxxx,
  "recall": 0.xxxx,
  "f1_macro": 0.xxxx,
  "confidence_interval_95": {
    "mean": 0.xxxx,
    "lower": 0.xxxx,
    "upper": 0.xxxx,
    "margin": 0.xxxx
  },
  "param_count": x.xx,
  "avg_inference_time": 0.xxx,
  "confusion_matrix": [[...]]
}
```

**comparison_metrics.json:**
```json
{
  "baseline_accuracy": 0.xxxx,
  "proposed_accuracy": 0.xxxx,
  "accuracy_improvement": 0.xxxx,
  "baseline_f1": 0.xxxx,
  "proposed_f1": 0.xxxx,
  "f1_improvement": 0.xxxx,
  "baseline_params": x.xx,
  "proposed_params": x.xx,
  "param_overhead": x.xx
}
```

### Ablation Results

**ablation_results.json:**
```json
{
  "E1_Full": {
    "description": "Full Dynamic VIC model",
    "vic_components": {...},
    "accuracy": 0.xxxx,
    "std": 0.xxxx,
    "f1_macro": 0.xxxx,
    "confidence_interval": {...}
  },
  ...
}
```

**component_importance.json:**
```json
{
  "Full Model Improvement": 0.xxxx,
  "Invariance Only": 0.xxxx,
  "Covariance Only": 0.xxxx,
  "Variance Only": 0.xxxx,
  "Dynamic Weight Effect": 0.xxxx
}
```

### McNemar's Test Results

**significance_tests.json:**
```json
{
  "pairwise_comparisons": [
    {
      "model_a": "E6_Baseline",
      "model_b": "E1_Full",
      "contingency_table": [n00, n01, n10, n11],
      "statistic": x.xxxx,
      "p_value": 0.xxxx,
      "significant_at_0.05": true/false,
      "effect_description": "..."
    },
    ...
  ]
}
```

### Feature Analysis Results

**feature_collapse_metrics.json:**
```json
{
  "baseline": {
    "collapse": {
      "collapsed_dimensions": x,
      "total_dimensions": x,
      "collapse_ratio": 0.xxxx,
      "min_std": 0.xxxxxx,
      "max_std": 0.xxxxxx,
      "mean_std": 0.xxxxxx
    },
    "utilization": {...},
    "comprehensive": {...}
  },
  "proposed": {...}
}
```

## Metrics Reported

### Quantitative Metrics

- **Accuracy** (mean ± std)
- **Precision** (macro-averaged)
- **Recall** (macro-averaged)
- **F1 Score** (macro)
- **95% Confidence Intervals**
- **Parameter Count** (millions)
- **Inference Time** (milliseconds)
- **Training Time** (seconds)
- **Confusion Matrix**

### Feature Analysis Metrics

- **Feature Collapse Detection**
  - Collapsed dimensions count and ratio
  - Standard deviation per dimension
  - Feature variance distribution
  
- **Feature Utilization**
  - Mean utilization score
  - Low utilization dimensions
  
- **Feature Redundancy**
  - High correlation pairs
  - Effective dimensions (PCA)
  - Dimensionality reduction ratio
  
- **Intra-class Consistency**
  - Euclidean consistency
  - Cosine consistency

### Statistical Testing

- **McNemar's Test**
  - Contingency tables
  - Chi-squared statistic
  - p-values
  - Significance levels
  - Effect descriptions

## Examples

### Example 1: Full Experiment on miniImagenet

```bash
python run_experiments.py \
  --dataset miniImagenet \
  --backbone Conv4 \
  --n_way 5 \
  --k_shot 1 \
  --num_epochs 50 \
  --test_iter 600 \
  --run_mode all \
  --output_dir ./results/mini_conv4
```

### Example 2: Ablation Study on CUB

```bash
python run_experiments.py \
  --dataset CUB \
  --backbone ResNet18 \
  --n_way 5 \
  --k_shot 5 \
  --run_mode ablation \
  --ablation_experiments E1,E2,E3,E4,E5,E6 \
  --output_dir ./results/cub_resnet18
```

### Example 3: Using Pre-trained Checkpoints

```bash
python run_experiments.py \
  --dataset miniImagenet \
  --backbone Conv4 \
  --baseline_checkpoint ./checkpoints/baseline_best.tar \
  --proposed_checkpoint ./checkpoints/proposed_best.tar \
  --run_mode qualitative \
  --output_dir ./results/mini_pretrained
```

### Example 4: Quick Feature Analysis

```bash
python run_experiments.py \
  --dataset CIFAR \
  --backbone Conv4 \
  --n_way 5 \
  --k_shot 1 \
  --test_iter 100 \
  --run_mode feature_analysis \
  --output_dir ./results/cifar_features
```

## Architecture

### Configuration Classes

The experiment runner uses dataclasses for configuration:

- **ExperimentConfig**: Main experiment configuration
- **AblationExperimentConfig**: Ablation experiment settings
- **VICComponents**: VIC component flags (invariance, covariance, variance, dynamic)
- **RunMode**: Enum for run modes

### Main Components

1. **Data Loading**: `get_data_loaders()`
2. **Model Creation**: `create_model()`
3. **Training**: `train_model()`
4. **Testing**: `test_model()`
5. **Qualitative Analysis**: `run_qualitative_analysis()`
6. **Ablation Study**: `run_ablation_study()`
7. **McNemar's Test**: `run_mcnemar_test()`
8. **Feature Analysis**: `run_feature_analysis()`

## Integration with Existing Code

The runner integrates with existing modules:

- **train.py**: Training loop logic
- **test.py**: Testing utilities
- **eval_utils.py**: Evaluation metrics
- **ablation_study.py**: McNemar's test implementation
- **feature_analysis.py**: Feature analysis utilities
- **qualitative_analysis.py**: Visualization utilities
- **methods/transformer.py**: Baseline model (FewShotTransformer)
- **methods/optimal_few_shot.py**: Proposed model (OptimalFewShot)

## Reproducibility

The runner ensures reproducibility through:

1. **Fixed random seeds** (configurable via `--seed`)
2. **Deterministic GPU operations** (cudnn.deterministic = True)
3. **Configuration logging** (all settings saved to JSON)
4. **Version tracking** (experiment timestamp)

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

```bash
# Reduce batch size by using fewer test iterations
python run_experiments.py --test_iter 100

# Use smaller backbone
python run_experiments.py --backbone Conv4

# Run modes separately
python run_experiments.py --run_mode train_test
python run_experiments.py --run_mode ablation
```

### Missing Dependencies

Install all required packages:

```bash
pip install torch torchvision numpy scipy matplotlib seaborn scikit-learn tqdm GPUtil psutil
```

### Dataset Not Found

Ensure datasets are properly configured in `configs.py`:

```python
data_dir['miniImagenet'] = '/path/to/dataset/miniImagenet/'
```

## Citation

If you use this unified experiment runner in your research, please cite:

```bibtex
@software{unified_experiment_runner,
  title={Unified Experiment Runner for Few-Shot Cosine Transformer},
  author={VCoklat},
  year={2024},
  url={https://github.com/VCoklat/Few-Shot-Cosine-Transformer}
}
```

## License

This project is licensed under the same license as the parent repository.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note:** This unified experiment runner is designed to work with the Few-Shot Cosine Transformer repository. Ensure all dependencies and datasets are properly configured before running experiments.
