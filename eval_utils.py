"""
Extended evaluation utilities for few-shot classification.
Tracks F1 plus a richer set of metrics and system stats.

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

# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader, model, n_way, class_names=None,
             chunk: int = 16, device: str = "cuda", track_all_classes: bool = True):
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
        all_classes_f1    – per-class F1 for all dataset classes (if track_all_classes=True)
        all_classes_names – names for all dataset classes (if track_all_classes=True)
    """
    model.eval()

    all_true, all_pred, all_scores, times = [], [], [], []
    
    # For tracking all classes across episodes
    all_true_global, all_pred_global = [], []
    
    # Get batch sampler to access episode class information
    batch_sampler = loader.batch_sampler if hasattr(loader, 'batch_sampler') else None
    dataset = loader.dataset if hasattr(loader, 'dataset') else None
    
    # Create an iterator for the batch sampler to track episode classes
    episode_iter = iter(batch_sampler) if (batch_sampler and track_all_classes) else None

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

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)

        preds = scores.argmax(1).numpy()
        all_pred.append(preds)
        all_scores.append(scores.numpy())

        n_query = len(preds) // n_way
        all_true.append(np.repeat(np.arange(n_way), n_query))
        
        # Track actual class labels if requested
        if episode_iter and dataset:
            try:
                sampled_class_indices = next(episode_iter)
                # Map predictions from 0..n_way-1 to actual class IDs
                actual_class_ids = [dataset.cl_list[idx] for idx in sampled_class_indices]
                
                # Map predictions and true labels to actual class IDs
                pred_global = np.array([actual_class_ids[p] for p in preds])
                true_global = np.repeat(actual_class_ids, n_query)
                
                all_pred_global.append(pred_global)
                all_true_global.append(true_global)
            except (StopIteration, AttributeError, IndexError) as e:
                # If we can't track episodes, disable tracking
                print(f"Warning: Could not track all classes: {e}")
                episode_iter = None

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
        if dataset and hasattr(dataset, 'class_labels') and hasattr(dataset, 'cl_list'):
            # Map class IDs to class names via cl_list
            all_classes_names = []
            for cls_id in all_class_ids:
                try:
                    # Find the index of this class ID in cl_list
                    idx = dataset.cl_list.index(cls_id)
                    all_classes_names.append(dataset.class_labels[idx])
                except (ValueError, IndexError):
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

    return res

# ──────────────────────────────────────────────────────────────
def pretty_print(res: dict) -> None:
    """Console-friendly summary of `evaluate()` output."""
    print("\n📊 EVALUATION RESULTS:")
    print("="*50)
    print(f"🎯 Macro-F1: {res['macro_f1']:.4f}")
    
    print(f"\n📈 Per-class F1 scores:")
    for i, f in enumerate(res["class_f1"]):
        name = res["class_names"][i] if i < len(res["class_names"]) else f"Class {i}"
        print(f"  F1 '{name}': {f:.4f}")

    print(f"\n🔢 Confusion matrix:")
    print(np.array(res["conf_mat"]))
    
    # Print all-classes F1 scores if available
    if 'all_classes_f1' in res:
        print(f"\n📊 F1 Scores for All Dataset Classes ({len(res['all_classes_f1'])} classes):")
        for i, (name, f1) in enumerate(zip(res['all_classes_names'], res['all_classes_f1'])):
            print(f"  {name}: {f1:.4f}")

    print(f"\n⏱️ Avg inference time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"💾 Model size: {res['param_count']:.2f} M params")
    print(f"🖥️ GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']:.1f}/{res['gpu_mem_total_MB']:.1f} MB")
    print(f"🖥️ CPU util: {res['cpu_util']:.1f}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")
    print("="*50)
