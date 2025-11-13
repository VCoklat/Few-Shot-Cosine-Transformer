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
             chunk: int = 16, device: str = "cuda"):
    """
    Evaluate `model` on an episodic `loader`.

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
    """
    model.eval()

    all_true, all_pred, all_scores, times = [], [], [], []

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

        torch.cuda.synchronize()
        times.append(time.time() - t0)

        preds = scores.argmax(1).numpy()
        all_pred.append(preds)
        all_scores.append(scores.numpy())

        n_query = len(preds) // n_way
        all_true.append(np.repeat(np.arange(n_way), n_query))

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

    res = dict(
        macro_f1        = float(f1_score(y_true, y_pred, average="macro")),
        class_f1        = f1_score(y_true, y_pred, average=None).tolist(),
        conf_mat        = confusion_matrix(y_true, y_pred).tolist(),
        accuracy        = accuracy_score(y_true, y_pred),
        macro_precision = macro_prec,
        macro_recall    = macro_rec,
        kappa           = cohen_kappa_score(y_true, y_pred),
        mcc             = matthews_corrcoef(y_true, y_pred),
        top5_accuracy   = top_k_accuracy_score(
                            y_true, y_scores, k=min(5, n_way), labels=list(range(n_way))
                          ),
        avg_inf_time    = float(np.mean(times)),
        param_count     = sum(p.numel() for p in model.parameters()) / 1e6,
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
    print(f"\nMacro-F1: {res['macro_f1']:.4f}")
    for name, f in zip(res["class_names"], res["class_f1"]):
        print(f"  F1 '{name}': {f:.4f}")

    print("\nConfusion matrix:")
    print(np.array(res["conf_mat"]))
    print(f"\nAccuracy:          {res['accuracy']:.4f}")
    print(f"Macro Precision:   {res['macro_precision']:.4f}")
    print(f"Macro Recall:      {res['macro_recall']:.4f}")
    print(f"Cohen's κ:         {res['kappa']:.4f}")
    print(f"Matthews CorrCoef: {res['mcc']:.4f}")
    print(f"Top-5 Accuracy:    {res['top5_accuracy']:.4f}")

    print(f"\nAvg inf. time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"Model size:            {res['param_count']:.2f} M params")
    print(f"GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']:.1f}/{res['gpu_mem_total_MB']:.1f} MB")
    print(f"CPU util: {res['cpu_util']:.1f}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")
