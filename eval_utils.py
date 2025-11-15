"""
Extended evaluation utilities for few-shot classification.
Tracks F1 plus a richer set of metrics and system stats.
Integrates comprehensive feature space analysis.

Author: dvh
"""

import gc
import time
import numpy as np
import torch
import psutil
import GPUtil

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
    top_k_accuracy_score,
)

try:
    from feature_analysis import comprehensive_feature_analysis, print_feature_analysis_summary
    FEATURE_ANALYSIS_AVAILABLE = True
except ImportError:
    FEATURE_ANALYSIS_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader, model, n_way, class_names=None,
             chunk: int = 16, device: str = "cuda", 
             extract_features: bool = False, feature_analysis: bool = False):
    """
    Evaluate `model` on an episodic `loader`.

    Args:
        loader: DataLoader with episodic batch sampler
        model: Model to evaluate
        n_way: Number of classes per episode
        class_names: Optional list of class names for all classes in dataset
        chunk: Chunk size for processing large batches
        device: Device to run evaluation on
        track_all_classes: If True, track and report F1 for all dataset classes.
                          If False, only report F1 for the n_way classes per episode.

    Returns a dict with:
        macro_f1          – overall F1 (macro)
        class_f1          – list of per-class F1
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
        episode_accuracies – per-episode accuracies (for confidence intervals)
        features          – extracted features (if extract_features=True)
        feature_analysis_results – feature analysis results (if feature_analysis=True)
    """
    model.eval()

    all_true, all_pred, all_scores, times = [], [], [], []
    episode_accuracies = []
    all_features = []

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
        if extract_features and hasattr(model, 'feature'):
            try:
                if x.size(0) > chunk:
                    feats = torch.cat(
                        [model.feature.forward(x[i:i + chunk].to(device)).cpu()
                         for i in range(0, x.size(0), chunk)],
                        dim=0
                    )
                else:
                    feats = model.feature.forward(x.to(device)).cpu()
                all_features.append(feats.numpy())
            except:
                pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)

        preds = scores.argmax(1).numpy()
        all_pred.append(preds)
        all_scores.append(scores.numpy())

        n_query = len(preds) // n_way
        y_true_episode = np.repeat(np.arange(n_way), n_query)
        all_true.append(y_true_episode)
        
        # Track per-episode accuracy
        episode_acc = np.mean(preds == y_true_episode)
        episode_accuracies.append(episode_acc)

        del scores
        gc.collect()
        torch.cuda.empty_cache()

    y_true   = np.concatenate(all_true)
    y_pred   = np.concatenate(all_pred)
    y_scores = np.concatenate(all_scores)

    # core classification metrics for episodic evaluation (n_way classes)
    macro_prec, macro_rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=list(range(n_way)), zero_division=0
    )

    res = dict(
        macro_f1        = float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_way)), zero_division=0)),
        class_f1        = f1_score(y_true, y_pred, average=None, labels=list(range(n_way)), zero_division=0).tolist(),
        conf_mat        = confusion_matrix(y_true, y_pred, labels=list(range(n_way))).tolist(),
        accuracy        = accuracy_score(y_true, y_pred),
        macro_precision = macro_prec,
        macro_recall    = macro_rec,
        kappa           = cohen_kappa_score(y_true, y_pred),
        mcc             = matthews_corrcoef(y_true, y_pred),
        top5_accuracy   = top_k_accuracy_score(
                            y_true, y_scores, k=5, labels=list(range(n_way))
                          ),
        avg_inf_time    = float(np.mean(times)),
        param_count     = sum(p.numel() for p in model.parameters()) / 1e6,
        episode_accuracies = episode_accuracies,
    )
    
    # Add all-classes F1 scores if we tracked them
    if all_true_global and all_pred_global:
        y_true_global = np.concatenate(all_true_global)
        y_pred_global = np.concatenate(all_pred_global)
        
        # Get all unique classes that appeared in episodes
        all_class_ids = np.unique(y_true_global)
        
        # Compute F1 scores for all classes
        all_classes_f1 = f1_score(y_true_global, y_pred_global, average=None, 
                                   labels=all_class_ids, zero_division=0)
        
        # Get class names if available
        if dataset and hasattr(dataset, 'class_labels'):
            # Map class IDs directly to class names (cls_id is the index in class_labels)
            all_classes_names = []
            for cls_id in all_class_ids:
                try:
                    # cls_id is the direct index into class_labels array
                    all_classes_names.append(dataset.class_labels[cls_id])
                except (IndexError):
                    all_classes_names.append(f"Class {cls_id}")
        else:
            all_classes_names = [f"Class {cls_id}" for cls_id in all_class_ids]
        
        res.update(
            all_classes_f1 = all_classes_f1.tolist(),
            all_classes_names = all_classes_names,
            all_class_ids = all_class_ids.tolist(),
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
        class_names       = class_names or list(range(len(res["class_f1"]))),
    )
    
    # Add feature analysis if requested
    if feature_analysis and FEATURE_ANALYSIS_AVAILABLE and len(all_features) > 0:
        print("\n🔬 Running comprehensive feature space analysis...")
        features = np.concatenate(all_features, axis=0)
        episode_accs_array = np.array(episode_accuracies)
        
        feature_results = comprehensive_feature_analysis(
            features=features,
            labels=y_true,
            episode_accuracies=episode_accs_array,
            predictions=y_pred
        )
        res['feature_analysis_results'] = feature_results
        res['features'] = features
        
        # Print summary
        print_feature_analysis_summary(feature_results)
    
    return res

# ──────────────────────────────────────────────────────────────
def evaluate_comprehensive(loader, model, n_way, class_names=None,
                          chunk: int = 16, device: str = "cuda",
                          feature_analysis: bool = True):
    """
    Wrapper for comprehensive evaluation combining standard metrics + feature analysis.
    
    Args:
        loader: Data loader
        model: Model to evaluate
        n_way: Number of classes
        class_names: Optional class names
        chunk: Batch size for chunked processing
        device: Device to use
        feature_analysis: Whether to perform feature analysis
        
    Returns:
        Dict with comprehensive evaluation results
    """
    return evaluate(
        loader=loader,
        model=model,
        n_way=n_way,
        class_names=class_names,
        chunk=chunk,
        device=device,
        extract_features=feature_analysis,
        feature_analysis=feature_analysis
    )

# ──────────────────────────────────────────────────────────────
def pretty_print(res: dict) -> None:
    """Console-friendly summary of `evaluate()` output with feature analysis support."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Classification Metrics
    print(f"\n📊 Classification Metrics:")
    print(f"  Accuracy:          {res['accuracy']:.4f}")
    print(f"  Macro-F1:          {res['macro_f1']:.4f}")
    print(f"  Macro Precision:   {res['macro_precision']:.4f}")
    print(f"  Macro Recall:      {res['macro_recall']:.4f}")
    print(f"  Cohen's κ:         {res['kappa']:.4f}")
    print(f"  Matthews CorrCoef: {res['mcc']:.4f}")
    print(f"  Top-5 Accuracy:    {res['top5_accuracy']:.4f}")
    
    # Per-class F1 Scores
    print(f"\n📈 Per-class F1 Scores:")
    for name, f in zip(res["class_names"], res["class_f1"]):
        print(f"  F1 '{name}': {f:.4f}")

    # Confusion Matrix
    print("\n🔢 Confusion Matrix:")
    print(np.array(res["conf_mat"]))
    
    # Statistical Confidence (if available from episode accuracies)
    if 'episode_accuracies' in res and len(res['episode_accuracies']) > 0:
        ep_accs = np.array(res['episode_accuracies'])
        mean_acc = np.mean(ep_accs)
        std_acc = np.std(ep_accs)
        n = len(ep_accs)
        margin = 1.96 * std_acc / np.sqrt(n)
        
        print(f"\n📊 Statistical Confidence (95% CI):")
        print(f"  Mean Episode Accuracy: {mean_acc*100:.2f}%")
        print(f"  Standard Deviation:    {std_acc*100:.2f}%")
        print(f"  95% Confidence Interval: [{(mean_acc-margin)*100:.2f}%, {(mean_acc+margin)*100:.2f}%]")
        print(f"  Number of Episodes:    {n}")

    # Performance Metrics
    print(f"\n⏱️ Performance:")
    print(f"  Avg inf. time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"  Model size:            {res['param_count']:.2f} M params")
    
    # Hardware Stats
    print(f"\n🖥️ Hardware Utilization:")
    print(f"  GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']:.0f}/{res['gpu_mem_total_MB']:.0f} MB")
    print(f"  CPU util: {res['cpu_util']:.1f}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")
    
    # Feature Analysis Results (if available)
    if 'feature_analysis_results' in res:
        print("\n" + "="*70)
        print("FEATURE ANALYSIS RESULTS (Already displayed above)")
        print("="*70)
    
    print("\n" + "="*70)
