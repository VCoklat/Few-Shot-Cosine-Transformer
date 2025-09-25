# train_test.py

# Main script for episodic few-shot training / testing

# ----------------------------------------------------

import os, random, time, pprint, gc, json
import numpy as np, torch, tqdm, wandb
import psutil, GPUtil
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, confusion_matrix

import backbone, configs
from data.datamgr import SetDataManager
from io_utils import model_dict, parse_args, get_best_file
from methods.transformer import FewShotTransformer
from methods.CTX import CTX

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
def change_model(name: str) -> str:
    """Swap vanilla Conv backbones for their 'NP' (no-pool) variants."""
    mapping = {"Conv4": "Conv4NP", "Conv6": "Conv6NP", "Conv4S": "Conv4SNP", "Conv6S": "Conv6SNP"}
    return mapping.get(name, name)

def seed_everything(seed=4040):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def evaluate(loader, model, n_way, class_names=None, chunk=16, device="cuda"):
    """Enhanced evaluation function with better error handling"""
    model.eval()
    all_true, all_pred, times = [], [], []

    for x, _ in loader:
        t0 = time.time()
        try:
            if x.size(0) > chunk:
                scores = torch.cat([
                    model.set_forward(x[i:i+chunk].to(device)).cpu()
                    for i in range(0, x.size(0), chunk)], 0)
            else:
                scores = model.set_forward(x.to(device)).cpu()

            torch.cuda.synchronize()
            times.append(time.time() - t0)

            preds = scores.argmax(1).numpy()
            all_pred.append(preds)

            num_per_class = len(preds) // n_way
            all_true.append(np.repeat(np.arange(n_way), num_per_class))

            del scores; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error in evaluation: {e}")
            continue

    if not all_true or not all_pred:
        print("Error: No valid predictions collected!")
        return {}

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    try:
        res = dict(
            macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            conf_mat = confusion_matrix(y_true, y_pred).tolist(),
            avg_inf_time = float(np.mean(times)) if times else 0.0,
            param_count = sum(p.numel() for p in model.parameters())/1e6
        )

        gpus = GPUtil.getGPUs()
        res.update(
            gpu_mem_used_MB = sum(g.memoryUsed for g in gpus) if gpus else 0,
            gpu_mem_total_MB = sum(g.memoryTotal for g in gpus) if gpus else 0,
            gpu_util = float(sum(g.load for g in gpus)/len(gpus)) if gpus else 0,
            cpu_util = psutil.cpu_percent(),
            cpu_mem_used_MB = psutil.virtual_memory().used / 1_048_576,
            cpu_mem_total_MB = psutil.virtual_memory().total / 1_048_576,
            class_names = class_names or list(range(len(res["class_f1"])))
        )

    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {}

    return res

def pretty_print(res):
    """Enhanced pretty printing"""
    if not res:
        print("No results to display!")
        return

    print(f"\n📊 EVALUATION RESULTS:")
    print("=" * 50)
    print(f"🎯 Macro-F1: {res['macro_f1']:.4f}")

    print("\n📈 Per-class F1 scores:")
    for name, f in zip(res["class_names"], res["class_f1"]):
        print(f"  F1 '{name}': {f:.4f}")

    print("\n🔢 Confusion matrix:")
    print(np.array(res["conf_mat"]))

    print(f"\n⏱️ Avg inference time/episode: {res['avg_inf_time']*1e3:.1f} ms")
    print(f"💾 Model size: {res['param_count']:.2f} M params")
    print(f"🖥️ GPU util: {res['gpu_util']*100:.1f}% | "
          f"mem {res['gpu_mem_used_MB']}/{res['gpu_mem_total_MB']} MB")
    print(f"🖥️ CPU util: {res['cpu_util']}% | "
          f"mem {res['cpu_mem_used_MB']:.0f}/{res['cpu_mem_total_MB']:.0f} MB")
    print("=" * 50)

def get_class_names_from_file(data_file, n_way=None):
    """Extract class names from JSON data file"""
    try:
        with open(data_file, 'r') as f:
            meta = json.load(f)

        unique_labels = np.unique(meta['image_labels']).tolist()

        if 'class_names' in meta:
            class_names = [meta['class_names'][str(label)] for label in unique_labels]
        else:
            class_names = [f"Class_{label}" for label in unique_labels]

        if n_way and len(class_names) > n_way:
            class_names = class_names[:n_way]

        return class_names
    except Exception as e:
        print(f"Error extracting class names: {e}")
        return [f"Class_{i}" for i in range(n_way)] if n_way else ["Class_0"]

def train_epoch(base_loader, model, optimizer, scaler, tqdm_bar):
    model.train()
    total_loss = 0

    for i, (x, _) in enumerate(base_loader):
        x = x.to(device)
        
        with autocast():
            loss = model.set_forward_loss(x)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        tqdm_bar.set_description(f'Loss: {loss.item():.4f}')
        tqdm_bar.update()
    
    return total_loss / len(base_loader)

def analyze_dynamic_weights(model):
    """Analyze the learned dynamic weights"""
    # Add to train.py after training loop
    for module in model.modules():
        if hasattr(module, 'record_weights'):
            module.record_weights = True  # Enable weight recording
    
    print("Collecting dynamic weight statistics...")
    with torch.no_grad():
        model.eval()
        with tqdm.tqdm(total=len(val_loader)) as pbar:
            for i, (x, _) in enumerate(val_loader):
                x = x.to(device)
                model.set_forward(x)
                pbar.update(1)
    
    # Run validation to collect weights
    for i, module in enumerate(model.modules()):
        if hasattr(module, 'get_weight_stats'):
            stats = module.get_weight_stats()
            if stats:
                print(f"Attention Block {i} weight stats:")
                if 'cosine_mean' in stats:  # 3-component format
                    print(f"  📊 Cosine weight stats: {stats['cosine_mean']:.4f} ± {stats['cosine_std']:.4f}")
                    print(f"  📊 Covariance weight stats: {stats['cov_mean']:.4f} ± {stats['cov_std']:.4f}")
                    print(f"  📊 Variance weight stats: {stats['var_mean']:.4f} ± {stats['var_std']:.4f}")
                    print("  📊 Distribution:")
                    for comp in ['cosine', 'cov', 'var']:
                        print(f"    {comp.capitalize()}:")
                        for bin_idx, count in enumerate(stats[f'histogram_{comp}']):
                            bin_start = bin_idx * 10
                            bin_end = (bin_idx + 1) * 10
                            print(f"      {bin_start:.1f}-{bin_end:.1f}: {count}")
                else:  # Legacy format
                    print(f"  📊 Mean: {stats['mean']:.4f}")
                    print(f"  📊 Std: {stats['std']:.4f}")
                    print(f"  📊 Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                    print("  📊 Distribution:")
                    for bin_idx, count in enumerate(stats['histogram']):
                        bin_start = bin_idx * 10
                        bin_end = (bin_idx + 1) * 10
                        print(f"    {bin_start:.1f}-{bin_end:.1f}: {count}")
            
            if hasattr(module, 'clear_weight_history'):
                module.clear_weight_history()
            if hasattr(module, 'record_weights'):
                module.record_weights = False

def direct_test(test_loader, model, params):
    acc = []
    iter_num = len(test_loader)
    
    with tqdm.tqdm(total=len(test_loader)) as pbar:
        for i, (x, _) in enumerate(test_loader):
            # Process batches in smaller chunks
            if x.size(0) > 16:  # If batch is larger than 16
                scores_list = []
                chunk_size = 16
                for j in range(0, x.size(0), chunk_size):
                    x_chunk = x[j:j+chunk_size].to(device)
                    with torch.no_grad():  # Ensure no gradients
                        score_chunk = model.set_forward(x_chunk)
                    scores_list.append(score_chunk.cpu())
                    torch.cuda.empty_cache()  # Clear cache after each chunk
                scores = torch.cat(scores_list, dim=0)
            else:
                with torch.no_grad():  # Ensure no gradients
                    x = x.to(device)
                    scores = model.set_forward(x)

            pred = scores.data.cpu().numpy().argmax(axis=1)
            y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
            acc.append(np.mean(pred == y)*100)
            
            pbar.set_description('Test Acc: {:.6f}'.format(np.mean(acc)))
            pbar.update(1)
    
    acc_all = np.asarray(acc)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    return acc_mean, acc_std

if __name__ == '__main__':
    params = parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(params))
    
    print('╭─' + '─'*50 + '─╮')
    print('│' + ' '*16 + 'Few-Shot-TransFormer' + ' '*15 + '│')
    print('╰─' + '─'*50 + '─╯')
    
    if params.dataset == 'Omniglot':
        params.n_query = 15
    
    if params.wandb:
        wandb_name = f"{params.method}_{params.backbone}_{params.dataset}_{params.n_way}w_{params.k_shot}s"
        if params.train_aug:
            wandb_name += "_aug"
        if params.FETI and "ResNet" in params.backbone:
            wandb_name += "_FETI"
        wandb_name += f"_{params.datetime}"
        
        wandb.init(project="Few-Shot-TransFormer", name=wandb_name, config=params, id=params.datetime)
    
    print('╭─' + '─'*50 + '─╮')
    print('│' + ' '*18 + 'DATASET LOADERS' + ' '*17 + '│')
    print('╰─' + '─'*50 + '─╯')
    
    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['Omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'
    
    image_size = 224 if "ResNet" in params.backbone else 84
    
    if params.dataset in ['Omniglot', 'cross_char']:
        if params.backbone == 'Conv4':
            params.backbone = 'Conv4S'
        if params.backbone == 'Conv6':
            params.backbone = 'Conv6S'
    
    optimization = params.optimization
    
    if params.method in ['FSCT-softmax', 'FSCT-cosine', 'CTX-softmax', 'CTX-cosine']:
        few_shot_params = dict(n_way=params.n_way, k_shot=params.k_shot, n_query=params.n_query)
        
        base_datamgr = SetDataManager(image_size, n_episode=params.n_episode, **few_shot_params)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        
        val_datamgr = SetDataManager(image_size, n_episode=params.n_episode, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        
        seed_func = seed_everything
        
        print('╭─' + '─'*50 + '─╮')
        print('│' + ' '*20 + 'BUILD MODEL' + ' '*19 + '│')
        print('╰─' + '─'*50 + '─╯')
        
        if params.method in ['FSCT-softmax', 'FSCT-cosine']:
            variant = "cosine" if params.method == "FSCT-cosine" else "softmax"
            
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](flatten=True) if "ResNet" in params.backbone else model_dict[params.backbone](flatten=True)
            
            model = FewShotTransformer(feature_model, variant=variant, **few_shot_params)
            
        elif params.method in ['CTX-softmax', 'CTX-cosine']:
            variant = "cosine" if params.method == "CTX-cosine" else "softmax"
            input_dim = 512 if "ResNet" in params.backbone else 64
            
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](flatten=False) if "ResNet" in params.backbone else model_dict[params.backbone](flatten=False)
            
            model = CTX(feature_model, variant=variant, input_dim=input_dim, **few_shot_params)
        else:
            raise ValueError("Unknown method")
        
        model = model.to(device)
        
        params.checkpoint_dir = '%s/%s/%s/%s' % (configs.save_dir, params.dataset, params.backbone, params.method)
        if params.train_aug:
            params.checkpoint_dir += '_aug'
        if params.FETI and "ResNet" in params.backbone:
            params.checkpoint_dir += '_FETI'
        
        params.checkpoint_dir += '_%dway_%dshot' % (params.n_way, params.k_shot)
        
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
        print('╭─' + '─'*50 + '─╮')
        print('│' + ' '*18 + 'TRAINING PHASE' + ' '*17 + '│')
        print('╰─' + '─'*50 + '─╯')
        
        # Training logic would go here
        # train(base_loader, val_loader, model, optimization, params.num_epoch, params)
        
        # Analyze weights
        # analyze_dynamic_weights(model)
        
        print('╭─' + '─'*50 + '─╮')
        print('│' + ' '*18 + 'TESTING PHASE' + ' '*18 + '│')
        print('╰─' + '─'*50 + '─╯')
        
        torch.cuda.empty_cache()
        
        split = params.split
        if params.dataset == 'cross':
            test_file = configs.data_dir['miniImagenet'] + 'all.json' if split == 'base' else configs.data_dir['CUB'] + f'{split}.json'
        elif params.dataset == 'cross_char':
            test_file = configs.data_dir['Omniglot'] + 'noLatin.json' if split == 'base' else configs.data_dir['emnist'] + f'{split}.json'
        else:
            test_file = configs.data_dir[params.dataset] + f'{split}.json'
        
        test_loader = SetDataManager(
            image_size,
            n_episode=params.test_iter,
            **dict(n_way=params.n_way, k_shot=params.k_shot, n_query=params.n_query)
        ).get_data_loader(test_file, aug=False)
        
        best = get_best_file(params.checkpoint_dir)
        if best:
            model.load_state_dict(torch.load(best)['state'])
        
        class_names = getattr(test_loader.dataset, "class_labels", None)
        metrics = evaluate(test_loader, model, params.n_way, class_names=class_names, device=device)
        pretty_print(metrics)
        
        if params.wandb:
            wandb.log(metrics)
            wandb.finish()
