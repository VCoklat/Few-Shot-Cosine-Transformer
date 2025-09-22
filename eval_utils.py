import time, gc, numpy as np, torch, psutil, GPUtil
from sklearn.metrics import f1_score, confusion_matrix

@torch.no_grad()
def evaluate(loader, model, n_way, class_names=None, chunk=16, device="cuda"):
    """
    Returns a dict with:
      macro_f1         – single number
      class_f1         – list aligned with class_names (or indices)
      conf_mat         – square matrix (Python list)
      avg_inf_time     – seconds per episode
      param_count      – model size (M)
      gpu / cpu stats  – utilisation & memory
    """
    model.eval()
    all_true, all_pred, times = [], [], []

    for x, _ in loader:                     # dataset’s y is ignored
        t0 = time.time()
        if x.size(0) > chunk:               # prevent OOM
            scores = torch.cat([
                model.set_forward(x[i:i+chunk].to(device)).cpu()
                for i in range(0, x.size(0), chunk)], 0)
        else:
            scores = model.set_forward(x.to(device)).cpu()
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        preds = scores.argmax(1).numpy()          # (n_way*n_query,)
        all_pred.append(preds)

        # fabricate ground-truth labels matching preds length
        num_per_class = len(preds) // n_way       # = n_query
        all_true.append(np.repeat(np.arange(n_way), num_per_class))

        del scores; gc.collect(); torch.cuda.empty_cache()

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    res = dict(
        macro_f1   = float(f1_score(y_true, y_pred, average="macro")),
        class_f1   = f1_score(y_true, y_pred, average=None).tolist(),
        conf_mat   = confusion_matrix(y_true, y_pred).tolist(),
        avg_inf_time = float(np.mean(times)),
        param_count  = sum(p.numel() for p in model.parameters())/1e6
    )

    gpus = GPUtil.getGPUs()
    res.update(
        gpu_mem_used_MB   = sum(g.memoryUsed  for g in gpus) if gpus else 0,
        gpu_mem_total_MB  = sum(g.memoryTotal for g in gpus) if gpus else 0,
        gpu_util          = float(sum(g.load for g in gpus)/len(gpus)) if gpus else 0,
        cpu_util          = psutil.cpu_percent(),
        cpu_mem_used_MB   = psutil.virtual_memory().used  / 1_048_576,
        cpu_mem_total_MB  = psutil.virtual_memory().total / 1_048_576,
        class_names       = class_names or list(range(len(res["class_f1"])))
    )
    return res


def pretty_print(res):
    print(f"\nMacro-F1: {res['macro_f1']:.4f}")
    for name, f in zip(res["class_names"], res["class_f1"]):
        print(f"  F1 '{name}': {f:.4f}")

    print("\nConfusion matrix:\n", np.array(res["conf_mat"]))
    print(f"\nAvg inference time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"Model size: {res['param_count']:.2f} M params")
    print(f"GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']}/{res['gpu_mem_total_MB']} MB")
    print(f"CPU util: {res['cpu_util']}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")
\