"""
Extended evaluation utilities for few-shot classification.
Tracks F1 plus a richer set of metrics and system stats.
Includes comprehensive feature analysis, confidence intervals, and McNemar's test.

Author: dvh
"""

import gc
import time
import numpy as np
import torch
import psutil
import GPUtil

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
    top_k_accuracy_score,
)

try:
    from feature_analysis import (
        compute_confidence_interval,
        comprehensive_feature_analysis
    )
    FEATURE_ANALYSIS_AVAILABLE = True
except ImportError:
    FEATURE_ANALYSIS_AVAILABLE = False
    print("Warning: feature_analysis module not available. Advanced metrics disabled.")

try:
    from ablation_study import (
        mcnemar_test,
        mcnemar_test_multiple,
        compute_contingency_table,
        format_contingency_table,
        AblationStudy,
        AblationConfig,
        AblationType
    )
    ABLATION_STUDY_AVAILABLE = True
except ImportError:
    ABLATION_STUDY_AVAILABLE = False
    print("Warning: ablation_study module not available. McNemar's test disabled.")

# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader, model, n_way, class_names=None,
             chunk: int = 16, device: str = "cuda", extract_features: bool = False):
    """
    Evaluate `model` on an episodic `loader`.

    Args:
        loader: DataLoader for episodic evaluation
        model: Model to evaluate
        n_way: Number of ways (classes per episode)
        class_names: Optional list of class names for display
        chunk: Batch size for chunked processing
        device: Device to run on
        extract_features: If True, also extract features for feature analysis

    Returns a dict with:
        conf_mat          – confusion matrix (list of lists)
        accuracy          – overall accuracy
        macro_precision   – macro-averaged precision
        macro_recall      – macro-averaged recall
        kappa             – Cohen's κ
        mcc               – Matthews Corr. Coef.
        top5_accuracy     – top-5 accuracy
        avg_inf_time      – mean inference time per episode (s)
        param_count       – model size (millions of params)
        gpu*/cpu*         – utilisation & memory stats
        class_names       – label strings for pretty print
        episode_accuracies – list of per-episode accuracies (for CI)
        features (optional) – extracted features if extract_features=True
    """
    model.eval()

    all_true, all_pred, all_scores, times = [], [], [], []
    episode_accuracies = []
    all_features = [] if extract_features else None

    for x, _ in loader:                     # dataset's y is ignored
        t0 = time.time()

        # forward pass in safe chunks
        if x.size(0) > chunk:
            scores = torch.cat(
                [model.set_forward(x[i:i + chunk].to(device)).cpu()
                 for i in range(0, x.size(0), chunk)],
                dim=0
            )
        else:
            scores = model.set_forward(x.to(device)).cpu()

        # Extract features if requested
        if extract_features:
            try:
                with torch.no_grad():
                    # Prefer parse_feature for meta-learning models (returns support/query split)
                    # as it handles episodic reshaping internally. Fallback to `feature` when not available.
                    if hasattr(model, 'parse_feature'):
                        try:
                            z_support, z_query = model.parse_feature(x, is_feature=False)
                            # Only use query features to match the labels (which are only for query samples)
                            feats = z_query.reshape(-1, z_query.size(-1)).cpu().numpy()
                        except Exception as e:
                            print(f"Warning: model.parse_feature failed: {e}")
                            feats = None
                    elif hasattr(model, 'feature'):
                        try:
                            # For models without parse_feature, extract all features
                            # Note: This assumes the model's feature() returns features for query samples only
                            feats = model.feature(x.to(device)).cpu().numpy()
                        except Exception as e:
                            print(f"Warning: model.feature failed: {e}")
                            feats = None
                    else:
                        feats = None

                if feats is not None:
                    all_features.append(feats)
                    # Debug: show feature extraction shape once
                    if len(all_features) == 1:
                        try:
                            print(f"Extracted features: shape={feats.shape}")
                        except Exception:
                            pass
            except Exception as e:
                # If feature extraction fails, continue without it
                pass

        torch.cuda.synchronize()
        times.append(time.time() - t0)

        preds = scores.argmax(1).numpy()
        all_pred.append(preds)
        all_scores.append(scores.numpy())

        n_query = len(preds) // n_way
        y_episode = np.repeat(np.arange(n_way), n_query)
        all_true.append(y_episode)
        
        # Compute episode accuracy
        episode_acc = np.mean(preds == y_episode)
        episode_accuracies.append(episode_acc)

        del scores
        gc.collect()
        torch.cuda.empty_cache()

    y_true   = np.concatenate(all_true)
    y_pred   = np.concatenate(all_pred)
    y_scores = np.concatenate(all_scores)

    # core classification metrics
    macro_prec, macro_rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Compute top-k accuracy, handling binary classification specially
    # sklearn's top_k_accuracy_score expects 1D scores for binary classification
    if n_way == 2:
        # For binary classification, use probability of positive class (class 1)
        top5_acc = top_k_accuracy_score(y_true, y_scores[:, 1], k=1)
    else:
        top5_acc = top_k_accuracy_score(
            y_true, y_scores, k=min(5, n_way), labels=list(range(n_way))
        )

    res = dict(
        conf_mat        = confusion_matrix(y_true, y_pred).tolist(),
        accuracy        = accuracy_score(y_true, y_pred),
        macro_precision = macro_prec,
        macro_recall    = macro_rec,
        kappa           = cohen_kappa_score(y_true, y_pred),
        mcc             = matthews_corrcoef(y_true, y_pred),
        top5_accuracy   = top5_acc,
        avg_inf_time    = float(np.mean(times)),
        param_count     = sum(p.numel() for p in model.parameters()) / 1e6,
        episode_accuracies = episode_accuracies,
    )

    # Compute 95% confidence interval
    if FEATURE_ANALYSIS_AVAILABLE and len(episode_accuracies) > 1:
        mean_acc, lower_ci, upper_ci = compute_confidence_interval(
            np.array(episode_accuracies), confidence=0.95
        )
        res.update(
            confidence_interval_95 = {
                'mean': float(mean_acc),
                'lower': float(lower_ci),
                'upper': float(upper_ci),
                'margin': float(mean_acc - lower_ci)
            }
        )

    # hardware stats
    gpus = GPUtil.getGPUs()
    res.update(
        gpu_mem_used_MB   = sum(g.memoryUsed for g in gpus) if gpus else 0,
        gpu_mem_total_MB  = sum(g.memoryTotal for g in gpus) if gpus else 0,
        gpu_util          = float(sum(g.load for g in gpus)/len(gpus)) if gpus else 0,
        cpu_util          = psutil.cpu_percent(),
        cpu_mem_used_MB   = psutil.virtual_memory().used  / 1_048_576,
        cpu_mem_total_MB  = psutil.virtual_memory().total / 1_048_576,
        # Class names fall back to the confusion matrix size
        class_names       = class_names or (list(range(len(res["conf_mat"]))) if "conf_mat" in res else []),
    )

    # Add features if extracted
    if extract_features and all_features:
        res['features'] = np.concatenate(all_features, axis=0)
        res['feature_labels'] = y_true

    return res

# ──────────────────────────────────────────────────────────────
def pretty_print(res: dict, show_feature_analysis: bool = False) -> None:
    """
    Console-friendly summary of `evaluate()` output.
    
    Args:
        res: Results dictionary from evaluate()
        show_feature_analysis: If True, display comprehensive feature analysis
    """
    print("\n" + "="*80)
    print("CLASSIFICATION METRICS")
    print("="*80)
    
    # Accuracy with confidence interval
    print(f"\nAccuracy: {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")
    if 'confidence_interval_95' in res:
        ci = res['confidence_interval_95']
        print(f"95% Confidence Interval: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        print(f"  (±{ci['margin']:.4f} or ±{ci['margin']*100:.2f}%)")
        print(f"  Based on {len(res['episode_accuracies'])} episodes")
    
    # Macro Precision/Recall
    print(f"\nMacro Precision: {res['macro_precision']:.4f}")
    print(f"Macro Recall: {res['macro_recall']:.4f}")
    
    # Additional metrics
    print(f"\nCohen's κ: {res['kappa']:.4f}")
    print(f"Matthews CorrCoef: {res['mcc']:.4f}")
    print(f"Top-5 Accuracy: {res['top5_accuracy']:.4f}")
    
    # Confusion Matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    conf_mat = np.array(res["conf_mat"])
    print(conf_mat)
    
    # Confusion matrix analysis
    print("\nConfusion Matrix Analysis:")
    for i, name in enumerate(res["class_names"]):
        total = conf_mat[i].sum()
        correct = conf_mat[i, i]
        if total > 0:
            print(f"  {name}: {correct}/{total} correct ({correct/total*100:.1f}%)")
    
    # System metrics
    print("\n" + "="*80)
    print("SYSTEM METRICS")
    print("="*80)
    print(f"Avg inference time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"Model size: {res['param_count']:.2f} M params")
    print(f"GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']:.0f}/{res['gpu_mem_total_MB']:.0f} MB")
    print(f"CPU util: {res['cpu_util']:.1f}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")
    
    # Feature analysis (if available)
    # Make sure the feature_analysis entry is present and not None
    if show_feature_analysis and res.get('feature_analysis'):
        print("\n" + "="*80)
        print("FEATURE SPACE ANALYSIS")
        print("="*80)
        
        fa = res['feature_analysis']
        
        # Feature collapse
        if 'feature_collapse' in fa:
            fc = fa['feature_collapse']
            print(f"\nFeature Collapse Detection:")
            print(f"  Collapsed dimensions: {fc['collapsed_dimensions']}/{fc['total_dimensions']} "
                  f"({fc['collapse_ratio']*100:.1f}%)")
            print(f"  Std range: [{fc['min_std']:.6f}, {fc['max_std']:.6f}], mean: {fc['mean_std']:.6f}")
        
        # Feature utilization
        if 'feature_utilization' in fa:
            fu = fa['feature_utilization']
            print(f"\nFeature Utilization:")
            print(f"  Mean utilization: {fu['mean_utilization']:.4f}")
            print(f"  Low utilization dims (<30%): {fu['low_utilization_dims']}")
        
        # Diversity score
        if 'diversity_score' in fa:
            ds = fa['diversity_score']
            print(f"\nDiversity Score:")
            print(f"  Mean diversity (CV): {ds['mean_diversity']:.4f}")
            print(f"  Std diversity: {ds['std_diversity']:.4f}")
        
        # Feature redundancy
        if 'feature_redundancy' in fa:
            fr = fa['feature_redundancy']
            print(f"\nFeature Redundancy:")
            print(f"  Total features: {fr['total_features']}")
            print(f"  Effective dimensions (95% variance): {fr['effective_dimensions_95pct']}")
            print(f"  Dimensionality reduction ratio: {fr['dimensionality_reduction_ratio']:.4f}")
            print(f"  High correlation pairs (>0.9): {fr['high_correlation_pairs']}")
            print(f"  Moderate correlation pairs (>0.7): {fr['moderate_correlation_pairs']}")
            print(f"  Mean absolute correlation: {fr['mean_abs_correlation']:.4f}")
        
        # Intra-class consistency
        if 'intraclass_consistency' in fa:
            ic = fa['intraclass_consistency']
            print(f"\nIntra-Class Consistency:")
            print(f"  Mean Euclidean consistency: {ic['mean_euclidean_consistency']:.4f}")
            print(f"  Mean Cosine consistency: {ic['mean_cosine_consistency']:.4f}")
            print(f"  Mean Combined consistency: {ic['mean_combined_consistency']:.4f}")
        
        # Confusing pairs
        if 'confusing_pairs' in fa:
            cp = fa['confusing_pairs']
            print(f"\nMost Confusing Class Pairs (closest centroids):")
            for pair in cp['most_confusing_pairs'][:5]:
                c1_name = res["class_names"][pair['class_1']] if pair['class_1'] < len(res["class_names"]) else f"Class {pair['class_1']}"
                c2_name = res["class_names"][pair['class_2']] if pair['class_2'] < len(res["class_names"]) else f"Class {pair['class_2']}"
                print(f"  {c1_name} ↔ {c2_name}: distance = {pair['distance']:.4f}")
            print(f"  Mean inter-centroid distance: {cp['mean_intercentroid_distance']:.4f}")
        
        # Imbalance ratio
        if 'imbalance_ratio' in fa:
            ir = fa['imbalance_ratio']
            print(f"\nClass Imbalance:")
            print(f"  Imbalance ratio: {ir['imbalance_ratio']:.4f}")
            print(f"  Min class samples: {ir['min_class_samples']}")
            print(f"  Max class samples: {ir['max_class_samples']}")
            print(f"  Mean class samples: {ir['mean_class_samples']:.1f} (±{ir['std_class_samples']:.1f})")
    
    print("\n" + "="*80)


# ──────────────────────────────────────────────────────────────
def evaluate_comprehensive(loader, model, n_way, class_names=None,
                           chunk: int = 16, device: str = "cuda"):
    """
    Comprehensive evaluation including feature space analysis.
    
    Args:
        loader: DataLoader for episodic evaluation
        model: Model to evaluate
        n_way: Number of ways (classes per episode)
        class_names: Optional list of class names for display
        chunk: Batch size for chunked processing
        device: Device to run on
    
    Returns:
        Dictionary with all metrics including feature analysis
    """
    # First, run standard evaluation with feature extraction
    res = evaluate(loader, model, n_way, class_names=class_names, 
                   chunk=chunk, device=device, extract_features=True)
    
    # If features were extracted, perform comprehensive feature analysis
    if FEATURE_ANALYSIS_AVAILABLE and 'features' in res and res['features'] is not None:
        try:
            features = res['features']
            labels = res['feature_labels']
            
            # Perform comprehensive feature analysis
            feature_analysis = comprehensive_feature_analysis(features, labels)
            res['feature_analysis'] = feature_analysis
            
            print("\n✓ Feature analysis completed successfully")
        except Exception as e:
            print(f"\n⚠ Feature analysis failed: {e}")
            res['feature_analysis'] = None
    else:
        if not FEATURE_ANALYSIS_AVAILABLE:
            print("\n⚠ Feature analysis module not available")
        else:
            print("\n⚠ Could not extract features for analysis")
            print("   Make sure your model exposes `feature()` or `parse_feature()` methods and that SciPy/scikit-learn are installed.")
        res['feature_analysis'] = None
    
    # Clean up large arrays from result to save memory
    if 'features' in res:
        del res['features']
    if 'feature_labels' in res:
        del res['feature_labels']
    
    return res


# ──────────────────────────────────────────────────────────────
# McNemar's Test Integration
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_predictions(loader, model, n_way, class_names=None,
                              chunk: int = 16, device: str = "cuda"):
    """
    Evaluate model and return predictions for McNemar's test comparison.
    
    This is similar to evaluate() but also returns the raw predictions
    and true labels needed for McNemar's statistical comparison.
    
    Args:
        loader: DataLoader for episodic evaluation
        model: Model to evaluate
        n_way: Number of ways (classes per episode)
        class_names: Optional list of class names for display
        chunk: Batch size for chunked processing
        device: Device to run on
    
    Returns:
        Tuple of (results_dict, predictions, true_labels)
    """
    model.eval()
    
    all_true, all_pred, all_scores, times = [], [], [], []
    episode_accuracies = []
    
    for x, _ in loader:
        t0 = time.time()
        
        if x.size(0) > chunk:
            scores = torch.cat(
                [model.set_forward(x[i:i + chunk].to(device)).cpu()
                 for i in range(0, x.size(0), chunk)],
                dim=0
            )
        else:
            scores = model.set_forward(x.to(device)).cpu()
        
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        
        preds = scores.argmax(1).numpy()
        all_pred.append(preds)
        all_scores.append(scores.numpy())
        
        n_query = len(preds) // n_way
        y_episode = np.repeat(np.arange(n_way), n_query)
        all_true.append(y_episode)
        
        episode_acc = np.mean(preds == y_episode)
        episode_accuracies.append(episode_acc)
        
        del scores
        gc.collect()
        torch.cuda.empty_cache()
    
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_scores = np.concatenate(all_scores)
    
    macro_prec, macro_rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    if n_way == 2:
        top5_acc = top_k_accuracy_score(y_true, y_scores[:, 1], k=1)
    else:
        top5_acc = top_k_accuracy_score(
            y_true, y_scores, k=min(5, n_way), labels=list(range(n_way))
        )
    
    res = dict(
        conf_mat=confusion_matrix(y_true, y_pred).tolist(),
        accuracy=accuracy_score(y_true, y_pred),
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        kappa=cohen_kappa_score(y_true, y_pred),
        mcc=matthews_corrcoef(y_true, y_pred),
        top5_accuracy=top5_acc,
        avg_inf_time=float(np.mean(times)),
        param_count=sum(p.numel() for p in model.parameters()) / 1e6,
        episode_accuracies=episode_accuracies,
    )
    
    if FEATURE_ANALYSIS_AVAILABLE and len(episode_accuracies) > 1:
        mean_acc, lower_ci, upper_ci = compute_confidence_interval(
            np.array(episode_accuracies), confidence=0.95
        )
        res.update(
            confidence_interval_95={
                'mean': float(mean_acc),
                'lower': float(lower_ci),
                'upper': float(upper_ci),
                'margin': float(mean_acc - lower_ci)
            }
        )
    
    gpus = GPUtil.getGPUs()
    res.update(
        gpu_mem_used_MB=sum(g.memoryUsed for g in gpus) if gpus else 0,
        gpu_mem_total_MB=sum(g.memoryTotal for g in gpus) if gpus else 0,
        gpu_util=float(sum(g.load for g in gpus)/len(gpus)) if gpus else 0,
        cpu_util=psutil.cpu_percent(),
        cpu_mem_used_MB=psutil.virtual_memory().used / 1_048_576,
        cpu_mem_total_MB=psutil.virtual_memory().total / 1_048_576,
        class_names=class_names or list(range(len(res["conf_mat"]))),
    )
    
    return res, y_pred, y_true


def compare_models_mcnemar(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    true_labels: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B"
) -> dict:
    """
    Compare two models using McNemar's test.
    
    McNemar's test determines if there is a statistically significant 
    difference between the error rates of two classifiers.
    
    Args:
        predictions_a: Predictions from first model
        predictions_b: Predictions from second model
        true_labels: True class labels
        model_a_name: Name for first model
        model_b_name: Name for second model
    
    Returns:
        Dictionary with McNemar's test results
    """
    if not ABLATION_STUDY_AVAILABLE:
        raise ImportError("ablation_study module not available. Please ensure it is installed.")
    
    result = mcnemar_test(predictions_a, predictions_b, true_labels)
    result['model_a_name'] = model_a_name
    result['model_b_name'] = model_b_name
    result['model_a_accuracy'] = float(np.mean(predictions_a == true_labels))
    result['model_b_accuracy'] = float(np.mean(predictions_b == true_labels))
    
    return result


def print_mcnemar_comparison(result: dict) -> None:
    """
    Print a formatted McNemar's test comparison result.
    
    Args:
        result: Result dictionary from compare_models_mcnemar
    """
    print("\n" + "=" * 80)
    print("McNEMAR'S TEST COMPARISON")
    print("=" * 80)
    
    model_a = result.get('model_a_name', 'Model A')
    model_b = result.get('model_b_name', 'Model B')
    
    print(f"\nComparing: {model_a} vs {model_b}")
    
    if 'model_a_accuracy' in result:
        print(f"\n{model_a} Accuracy: {result['model_a_accuracy']:.4f}")
    if 'model_b_accuracy' in result:
        print(f"{model_b} Accuracy: {result['model_b_accuracy']:.4f}")
    
    print(f"\nContingency Table:")
    n00, n01, n10, n11 = result['contingency_table']
    print(f"  Both correct: {n11}")
    print(f"  Both wrong: {n00}")
    print(f"  {model_a} correct, {model_b} wrong: {n10}")
    print(f"  {model_a} wrong, {model_b} correct: {n01}")
    
    print(f"\nDiscordant pairs: {result['discordant_pairs']}")
    print(f"Test type: {result['test_type']}")
    print(f"Test statistic: {result['statistic']:.4f}")
    print(f"P-value: {result['p_value']:.6f}")
    
    print(f"\nResult: {result['effect_description']}")
    
    if result['significant_at_0.05']:
        if result['algorithm_a_better']:
            print(f"✓ {model_a} performs significantly better than {model_b}")
        elif result['algorithm_b_better']:
            print(f"✓ {model_b} performs significantly better than {model_a}")
    else:
        print("○ No statistically significant difference detected")
    
    print("=" * 80)


def run_ablation_comparison(
    loader,
    models: list,
    model_names: list,
    n_way: int,
    device: str = "cuda",
    baseline_index: int = 0
) -> dict:
    """
    Run ablation comparison across multiple models using McNemar's test.
    
    Args:
        loader: DataLoader for evaluation
        models: List of models to compare
        model_names: Names for each model
        n_way: Number of ways (classes per episode)
        device: Device to run on
        baseline_index: Index of the baseline model for comparisons
    
    Returns:
        Dictionary with all comparison results
    """
    if not ABLATION_STUDY_AVAILABLE:
        raise ImportError("ablation_study module not available.")
    
    if len(models) != len(model_names):
        raise ValueError("Number of models must match number of names")
    
    print(f"\nRunning ablation comparison across {len(models)} models...")
    
    # Evaluate all models
    all_results = []
    all_predictions = []
    true_labels = None
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\nEvaluating {name}...")
        res, preds, labels = evaluate_with_predictions(
            loader, model, n_way, device=device
        )
        all_results.append(res)
        all_predictions.append(preds)
        
        if true_labels is None:
            true_labels = labels
        
        print(f"  Accuracy: {res['accuracy']:.4f}")
    
    # Create ablation study
    study = AblationStudy(baseline_name=model_names[baseline_index])
    
    for i, (name, res, preds) in enumerate(zip(model_names, all_results, all_predictions)):
        config = AblationConfig(
            name=name,
            ablation_type=AblationType.FULL_MODEL if i == baseline_index else AblationType.CUSTOM,
            description=f"Model configuration: {name}"
        )
        
        ci = res.get('confidence_interval_95', {})
        ci_tuple = (ci.get('lower', 0.0), ci.get('upper', 0.0))
        
        study.add_result(
            name=name,
            config=config,
            accuracy=res['accuracy'],
            std=np.std(res['episode_accuracies']),
            predictions=preds,
            true_labels=true_labels,
            confidence_interval=ci_tuple,
            additional_metrics=res,
            is_baseline=(i == baseline_index)
        )
    
    # Generate and print report
    print("\n" + study.generate_report())
    
    # Perform pairwise McNemar's tests
    pairwise_results = mcnemar_test_multiple(
        all_predictions, model_names, true_labels
    )
    
    return {
        'individual_results': {name: res for name, res in zip(model_names, all_results)},
        'ablation_study': study,
        'pairwise_mcnemar': pairwise_results,
        'baseline': model_names[baseline_index]
    }
