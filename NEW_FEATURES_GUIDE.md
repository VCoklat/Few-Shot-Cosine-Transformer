# New Features: Plot Display and McNemar Testing

This document describes the new features added to the Unified Experiment Runner.

## Overview

Two new command-line flags have been added to enhance the experiment workflow:

1. **`--show_plots`**: Display visualizations interactively during execution
2. **`--mcnemar_each_test`**: Run McNemar statistical significance testing after each testing phase

## Feature 1: Interactive Plot Display (`--show_plots`)

### Description
By default, all plots (confusion matrices, t-SNE visualizations, ablation comparisons, etc.) are saved to disk but not displayed interactively. The `--show_plots` flag enables interactive display of all visualizations using `matplotlib.pyplot.show()`.

### Usage

```bash
# Enable plot display
python run_experiments.py --dataset miniImagenet --backbone Conv4 --show_plots

# Combined with other options
python run_experiments.py --dataset CUB --backbone ResNet18 --run_mode qualitative --show_plots
```

### Default Behavior
- **Default**: `False` (plots are saved but not displayed)
- **When enabled**: Each plot will be displayed in a window before the script continues

### Affected Visualizations
- Confusion matrices (baseline and proposed models)
- t-SNE feature visualizations
- Ablation study comparison plots
- Component importance analysis
- McNemar's test p-value matrices
- Contingency tables
- Feature variance distributions
- Feature correlation matrices

### Implementation Details
The `safe_plot_save()` function now accepts a `show` parameter:

```python
def safe_plot_save(output_path: str, dpi: int = 150, show: bool = False):
    """
    Safely save and close a matplotlib plot with proper exception handling.
    
    Args:
        output_path: Path to save the plot
        dpi: DPI for the saved image
        show: Whether to display the plot interactively before closing
    """
    try:
        plt.savefig(output_path, dpi=dpi)
        logger.info(f"  Saved: {output_path}")
        if show:
            plt.show()
    except Exception as e:
        logger.error(f"  Failed to save plot to {output_path}: {e}")
    finally:
        plt.close()
```

## Feature 2: McNemar Testing During Each Phase (`--mcnemar_each_test`)

### Description
McNemar's statistical test is used to determine if there's a statistically significant difference between two classification algorithms. Previously, this test was only available as a separate phase after ablation studies. Now, with `--mcnemar_each_test`, you can run significance testing immediately after each model comparison.

### Usage

```bash
# Enable McNemar testing after each test
python run_experiments.py --dataset miniImagenet --backbone Conv4 --mcnemar_each_test

# Combined with train_test mode
python run_experiments.py --dataset CUB --backbone ResNet18 --run_mode train_test --mcnemar_each_test

# Combined with ablation study
python run_experiments.py --dataset miniImagenet --backbone Conv4 --run_mode ablation --mcnemar_each_test
```

### Default Behavior
- **Default**: `False` (McNemar testing only in dedicated phase)
- **When enabled**: McNemar tests run automatically after each model comparison

### When McNemar Tests Are Run

#### 1. During Train/Test Phase (`run_train_test`)
After both baseline and proposed models are tested, McNemar's test compares their predictions:

```
Baseline vs Proposed
  - p-value
  - Significance level
  - Effect description
  - Contingency table
```

#### 2. During Ablation Study (`run_ablation_study`)
After each ablation experiment is tested (except E6 baseline), McNemar's test compares it against E6 (baseline):

```
E1_Full vs E6_Baseline
E2_InvDyn vs E6_Baseline
E3_InvCovDyn vs E6_Baseline
... and so on
```

### Output Format

```
============================================================
McNEMAR'S TEST: Baseline vs Proposed
============================================================
  p-value: 0.002145
  Significant at 0.05: True
  Effect: Algorithm B significantly outperforms A (significant (p < 0.05))
  Discordant pairs: 147
  Contingency table:
    Both correct: 2453
    Both wrong: 127
    Baseline correct, Proposed wrong: 58
    Baseline wrong, Proposed correct: 89
============================================================
```

### Implementation Details

A new helper function `run_mcnemar_comparison()` performs the test and displays results:

```python
def run_mcnemar_comparison(preds_a, preds_b, labels, name_a, name_b):
    """
    Run and display McNemar's test comparison between two models.
    
    Args:
        preds_a: Predictions from model A
        preds_b: Predictions from model B
        labels: True labels
        name_a: Name of model A
        name_b: Name of model B
    
    Returns:
        McNemar test result dictionary or None if test not available
    """
    # ... implementation details ...
```

### Saved Results
When `--mcnemar_each_test` is enabled, results are saved to:

- **Train/Test Phase**: `{output_dir}/quantitative/mcnemar_test.json`
- **Ablation Study**: Included in ablation results JSON with key `mcnemar_vs_baseline`

## Configuration

### In Code
```python
from configs.experiment_config import ExperimentConfig, RunMode

config = ExperimentConfig(
    dataset='miniImagenet',
    backbone='Conv4',
    show_plots=True,
    mcnemar_each_test=True,
    run_mode=RunMode.ALL
)
```

### Via Command Line
```bash
python run_experiments.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --show_plots \
    --mcnemar_each_test \
    --run_mode all
```

## Examples

### Example 1: Full Experiment with Both Features
```bash
python run_experiments.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --n_way 5 \
    --k_shot 1 \
    --show_plots \
    --mcnemar_each_test \
    --run_mode all
```

This will:
1. Train and test baseline and proposed models
2. Run McNemar's test comparing baseline vs proposed
3. Display all plots interactively
4. Generate qualitative analysis with displayed plots
5. Run ablation study with McNemar tests for each configuration
6. Run dedicated McNemar phase
7. Perform feature analysis with displayed plots

### Example 2: Quick Test with Significance Analysis
```bash
python run_experiments.py \
    --dataset CUB \
    --backbone ResNet18 \
    --run_mode train_test \
    --mcnemar_each_test \
    --baseline_checkpoint ./checkpoints/baseline.tar \
    --proposed_checkpoint ./checkpoints/proposed.tar
```

This will:
1. Load pretrained models
2. Test both models
3. Run McNemar's test immediately
4. Save results without full ablation study

### Example 3: Ablation Study with Immediate Feedback
```bash
python run_experiments.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --run_mode ablation \
    --mcnemar_each_test \
    --ablation_experiments E1,E2,E3,E6
```

This will:
1. Run specified ablation experiments
2. After each experiment (except E6), immediately compare to E6 baseline using McNemar's test
3. Show statistical significance results as soon as each configuration is tested

## Benefits

### `--show_plots`
- **Immediate visual feedback** during long experiments
- **Interactive exploration** of visualizations
- **Quick quality checks** without opening saved files
- **Easier debugging** of visualization issues

### `--mcnemar_each_test`
- **Earlier insights** into statistical significance
- **Immediate validation** of improvements
- **Better experiment monitoring** during long runs
- **No need to wait** for dedicated McNemar phase
- **Confidence in results** as experiments progress

## Testing

Unit tests are available in:
- `test_new_features.py` - Tests for configuration flags
- `test_integration_new_features.py` - Integration tests with command-line parsing

Run tests:
```bash
python test_new_features.py
python test_integration_new_features.py
```

## Technical Notes

1. **Plot Display**: When `show_plots=True`, the script will pause at each plot until you close the window. This is standard matplotlib behavior.

2. **McNemar Dependencies**: McNemar testing requires the `ablation_study` module with scipy. If not available, a warning is logged and the test is skipped.

3. **Performance**: Enabling `--mcnemar_each_test` adds minimal overhead as it only compares predictions (no model re-execution).

4. **Output Files**: Both features save results to disk regardless of the flags. The flags only control *additional* behaviors (display and inline testing).

## Backward Compatibility

These features are **fully backward compatible**:
- Default values are `False` for both flags
- Existing scripts and workflows continue to work without modification
- No changes to saved output file formats or locations
- All existing functionality remains unchanged
