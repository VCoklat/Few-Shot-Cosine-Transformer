# Analysis Scripts for Enhanced Few-Shot Learning

This directory contains scripts for qualitative and quantitative analysis of the enhanced few-shot learning model.

## Scripts

### 1. Qualitative Analysis (`qualitative_analysis.py`)

Generates visualizations of feature embeddings and class separation.

**Features:**
- **t-SNE Visualization**: 2D and 3D projections of feature space
- **PCA Analysis**: Principal component analysis with variance explained
- **Embedding Space Analysis**: Multi-view analysis including:
  - Class-wise feature distribution
  - Feature magnitude distribution per class
  - Inter-class distance matrix
  - Intra-class variance per class

**Usage:**
```bash
python qualitative_analysis.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --checkpoint ./checkpoint_models/best_model.tar \
    --n_way 5 \
    --k_shot 1 \
    --n_episodes 100 \
    --output_dir ./qualitative_results \
    --use_task_invariance 1 \
    --use_multi_scale 1 \
    --use_feature_augmentation 1 \
    --domain general
```

**Parameters:**
- `--dataset`: Dataset name (Omniglot/miniImagenet/CUB/HAM10000)
- `--backbone`: Backbone architecture (Conv4/ResNet18/ResNet34)
- `--checkpoint`: Path to trained model checkpoint (required)
- `--n_way`: Number of classes per episode (default: 5)
- `--k_shot`: Number of support samples per class (default: 1)
- `--n_query`: Number of query samples per class (default: 16)
- `--n_episodes`: Number of episodes to analyze (default: 100)
- `--output_dir`: Directory for output plots (default: ./qualitative_results)
- `--perplexity`: t-SNE perplexity parameter (default: 30)
- `--use_task_invariance`: Enable task-adaptive invariance (0/1, default: 1)
- `--use_multi_scale`: Enable multi-scale invariance (0/1, default: 1)
- `--use_feature_augmentation`: Enable feature augmentation (0/1, default: 1)
- `--use_prototype_refinement`: Enable prototype refinement (0/1, default: 0)
- `--domain`: Domain type (general/medical/fine_grained, default: general)

**Outputs:**
- `tsne_2d.png`: 2D t-SNE visualization of feature embeddings
- `tsne_3d.png`: 3D t-SNE visualization of feature embeddings
- `pca.png`: PCA projection with variance explained
- `embedding_analysis.png`: Comprehensive 4-panel analysis

**Example Output Description:**

1. **t-SNE plots**: Show how well classes are separated in the learned feature space. Well-separated clusters indicate good feature learning.

2. **PCA plot**: Shows the top 2 principal components and their explained variance ratio.

3. **Embedding Analysis** (4 panels):
   - Top-left: PCA projection colored by class
   - Top-right: Feature magnitude distribution per class
   - Bottom-left: Inter-class distance matrix (higher = better separation)
   - Bottom-right: Intra-class variance per class (lower = more compact)

---

### 2. F1 Score Evaluation (`evaluate_f1_scores.py`)

Computes detailed classification metrics including F1 scores for all classes.

**Features:**
- Per-class F1 scores, precision, and recall
- Overall metrics (macro, micro, weighted averaging)
- Confusion matrix
- Detailed classification report
- JSON output for programmatic access

**Usage:**
```bash
python evaluate_f1_scores.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --checkpoint ./checkpoint_models/best_model.tar \
    --n_way 5 \
    --k_shot 1 \
    --n_episodes 600 \
    --output_dir ./evaluation_results \
    --use_task_invariance 1 \
    --use_multi_scale 1 \
    --use_feature_augmentation 1 \
    --domain general
```

**Parameters:**
Same as qualitative analysis, with key differences:
- `--n_episodes`: Number of test episodes (default: 600, higher = more reliable metrics)
- `--output_dir`: Directory for results (default: ./evaluation_results)

**Outputs:**

**Console Output:**
```
================================================================================
EVALUATION RESULTS
================================================================================

Overall Accuracy: 65.34% ± 1.23%
  (Mean ± 95% CI over episodes)

--------------------------------------------------------------------------------
Overall Metrics:
--------------------------------------------------------------------------------
  F1 Score (Macro):    0.6512
  F1 Score (Micro):    0.6534
  F1 Score (Weighted): 0.6523
  Precision (Macro):   0.6589
  Recall (Macro):      0.6534

--------------------------------------------------------------------------------
Per-Class Metrics:
--------------------------------------------------------------------------------
Class      F1 Score     Precision    Recall      
--------------------------------------------------------------------------------
Class 0    0.6723       0.6801       0.6645
Class 1    0.6412       0.6523       0.6302
Class 2    0.6589       0.6634       0.6544
Class 3    0.6451       0.6578       0.6325
Class 4    0.6385       0.6409       0.6361
--------------------------------------------------------------------------------
Average    0.6512       0.6589       0.6534

--------------------------------------------------------------------------------
Confusion Matrix:
--------------------------------------------------------------------------------
Pred/True   Class 0    Class 1    Class 2    Class 3    Class 4    
--------------------------------------------------------------------------------
Class 0     1589       45         38         42         36        
Class 1     53         1512       49         55         51        
Class 2     41         52         1569       48         40        
Class 3     48         58         45         1519       50        
Class 4     39         53         39         56         1533      
================================================================================
```

**JSON File** (`f1_scores_<dataset>_<n_way>way_<k_shot>shot.json`):
```json
{
  "accuracy": 0.6534,
  "accuracy_std": 0.0623,
  "accuracy_95ci": 0.0123,
  "f1_macro": 0.6512,
  "f1_micro": 0.6534,
  "f1_weighted": 0.6523,
  "precision_macro": 0.6589,
  "recall_macro": 0.6534,
  "f1_per_class": {
    "Class_0": 0.6723,
    "Class_1": 0.6412,
    ...
  },
  "confusion_matrix": [[1589, 45, 38, 42, 36], ...],
  "classification_report": { ... }
}
```

---

## Example Workflows

### Complete Analysis Pipeline

```bash
# 1. Train model
python train_enhanced.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --k_shot 1 \
    --num_epoch 100

# 2. Generate qualitative visualizations
python qualitative_analysis.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --checkpoint ./checkpoint_models/best_model.tar \
    --n_episodes 100

# 3. Evaluate with F1 scores
python evaluate_f1_scores.py \
    --dataset miniImagenet \
    --backbone Conv4 \
    --checkpoint ./checkpoint_models/best_model.tar \
    --n_episodes 600
```

### Analyzing Different Configurations

Compare models with/without invariance modules:

```bash
# Baseline (no invariance modules)
python evaluate_f1_scores.py \
    --checkpoint baseline.tar \
    --use_task_invariance 0 \
    --use_multi_scale 0 \
    --use_feature_augmentation 0 \
    --output_dir ./results_baseline

# With all modules
python evaluate_f1_scores.py \
    --checkpoint enhanced.tar \
    --use_task_invariance 1 \
    --use_multi_scale 1 \
    --use_feature_augmentation 1 \
    --output_dir ./results_enhanced
```

### Medical Imaging Analysis (HAM10000)

```bash
python qualitative_analysis.py \
    --dataset HAM10000 \
    --backbone Conv4 \
    --checkpoint ./ham10000_model.tar \
    --n_way 7 \
    --k_shot 5 \
    --domain medical \
    --use_task_invariance 1 \
    --use_multi_scale 1 \
    --output_dir ./ham10000_analysis

python evaluate_f1_scores.py \
    --dataset HAM10000 \
    --backbone Conv4 \
    --checkpoint ./ham10000_model.tar \
    --n_way 7 \
    --k_shot 5 \
    --domain medical \
    --n_episodes 600 \
    --output_dir ./ham10000_evaluation
```

---

## Interpreting Results

### Qualitative Analysis

**Good indicators:**
- Well-separated clusters in t-SNE/PCA plots
- High inter-class distances (heatmap)
- Low intra-class variance
- Uniform feature magnitude distribution

**Poor indicators:**
- Overlapping clusters
- Low inter-class distances
- High intra-class variance
- Highly variable feature magnitudes

### F1 Score Evaluation

**Understanding metrics:**
- **F1 Score**: Harmonic mean of precision and recall (0-1, higher is better)
- **Macro**: Unweighted mean across classes (treats all classes equally)
- **Micro**: Global average (weighted by support)
- **Weighted**: Weighted average by class support

**Confusion Matrix:**
- Diagonal elements: Correct predictions
- Off-diagonal: Misclassifications
- Look for systematic errors (which classes confuse each other)

**Per-class analysis:**
- Classes with low F1: May need more training data or domain-specific modules
- High precision, low recall: Model is conservative
- Low precision, high recall: Model is aggressive

---

## Requirements

Both scripts require:
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tqdm

Install with:
```bash
pip install torch numpy scikit-learn matplotlib seaborn tqdm
```

---

## Tips

1. **t-SNE perplexity**: Adjust based on dataset size. Typical range: 5-50.
2. **Number of episodes**: More episodes = more reliable metrics (600+ recommended for final evaluation)
3. **Checkpoint loading**: Scripts handle both `model_state_dict` and `state` keys
4. **GPU usage**: Scripts automatically use CUDA if available
5. **Memory**: Qualitative analysis loads all features in memory. Reduce `n_episodes` if OOM occurs.

---

## Troubleshooting

**Issue: "Cannot load checkpoint"**
- Ensure checkpoint path is correct
- Check that model configuration matches training (n_way, k_shot, etc.)

**Issue: "Out of memory during t-SNE"**
- Reduce `n_episodes` for qualitative analysis
- Use `perplexity=5` for faster computation

**Issue: "Inconsistent metrics"**
- Increase `n_episodes` for evaluation (600+ recommended)
- Ensure test data is properly shuffled

**Issue: "Poor class separation in t-SNE"**
- This may indicate insufficient training
- Try different perplexity values (5, 10, 30, 50)
- Check if model is in eval mode (scripts handle this automatically)
