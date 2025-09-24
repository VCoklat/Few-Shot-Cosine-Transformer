# train_test.py
# Main script for episodic few-shot training / testing
# ----------------------------------------------------

import os, random, time, pprint
import numpy as np, torch, tqdm, wandb
from torch.cuda.amp import autocast, GradScaler

import backbone, configs
from data.datamgr import SetDataManager
from io_utils import model_dict, parse_args, get_best_file
from methods.transformer import FewShotTransformer
from methods.CTX import CTX
from eval_utils import evaluate, pretty_print

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
def change_model(name: str) -> str:
    """Swap vanilla Conv backbones for their ‘NP’ (no-pool) variants."""
    mapping = {"Conv4": "Conv4NP", "Conv6": "Conv6NP", "Conv4S": "Conv4SNP", "Conv6S": "Conv6SNP"}
    return mapping.get(name, name)


def seed_everything(seed=4040):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def change_model(model_name):
    if model_name == 'Conv4':
        model_name = 'Conv4NP'
    elif model_name == 'Conv6':
        model_name = 'Conv6NP'
    elif model_name == 'Conv4S':
        model_name = 'Conv4SNP'
    elif model_name == 'Conv6S':
        model_name = 'Conv6SNP'
    return model_name

if __name__ == '__main__':
    
    params = parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(params))
    print()
        
    project_name = "Few-Shot_TransFormer"            
    
    if params.dataset == 'Omniglot': params.n_query = 15
    if params.wandb:

        wandb_name = params.method + "_" + params.backbone + "_" + params.dataset + \
            "_" + str(params.n_way) + "w" + str(params.k_shot) + "s"
        
        if params.train_aug:
            wandb_name += "_aug"
        if params.FETI and 'ResNet' in params.backbone:
            wandb_name += "_FETI"
        wandb_name += "_" + params.datetime
        
        wandb.init(project=project_name, name=wandb_name,
                   config=params, id=params.datetime)
        print()

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['Omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if params.dataset == "CIFAR":
        image_size = 112 if 'ResNet' in params.backbone else 64
    else:
        image_size = 224 if 'ResNet' in params.backbone else 84

    if params.dataset in ['Omniglot', 'cross_char']:
        if params.backbone == 'Conv4': params.backbone = 'Conv4S'
        if params.backbone == 'Conv6': params.backbone = 'Conv6S'

    optimization = params.optimization

    if params.method in ['FSCT_softmax', 'FSCT_cosine', 'CTX_softmax', 'CTX_cosine']:

        few_shot_params = dict(
            n_way=params.n_way, k_shot=params.k_shot, n_query = params.n_query)
        base_datamgr = SetDataManager(
            image_size, n_episode=params.n_episode, **few_shot_params)
        base_loader = base_datamgr.get_data_loader(
            base_file, aug=params.train_aug)

        val_datamgr = SetDataManager(
            image_size, n_episode=params.n_episode, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(
            val_file, aug=False)
       
        seed_func()

        if params.method in ['FSCT_softmax', 'FSCT_cosine']:
            variant = 'cosine' if params.method == 'FSCT_cosine' else 'softmax'
            
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=True) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=True)

            model = FewShotTransformer(feature_model, variant=variant, **few_shot_params)
            
        elif params.method in ['CTX_softmax', 'CTX_cosine']:
            variant = 'cosine' if params.method == 'CTX_cosine' else 'softmax'
            input_dim = 512 if "ResNet" in params.backbone else 64
            def feature_model():
                if params.dataset in ['Omniglot', 'cross_char']:
                    params.backbone = change_model(params.backbone)
                return model_dict[params.backbone](params.FETI, params.dataset, flatten=False) if 'ResNet' in params.backbone else model_dict[params.backbone](params.dataset, flatten=False)

            model = CTX(feature_model, variant=variant, input_dim=input_dim, **few_shot_params)
    else:
        raise ValueError('Unknown method')

    model = model.to(device)
    
    
    params.checkpoint_dir = '%sc/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.backbone, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if params.FETI and 'ResNet' in params.backbone:
        params.checkpoint_dir += '_FETI'

    params.checkpoint_dir += '_%dway_%dshot' % (
        params.n_way, params.k_shot)
    
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    
    print("===================================")
    print("Train phase: ")
    
    model = train(base_loader, val_loader,  model, optimization, params.num_epoch, params)

    # Add to train.py after training loop
    def analyze_dynamic_weights(model):
        """Analyze the learned dynamic weights"""
        # Enable weight recording
        for module in model.modules():
            if isinstance(module, Attention):
                module.record_weights = True
        
        # Run validation to collect weights
        print("Collecting dynamic weight statistics...")
        with torch.no_grad():
            model.eval()
            with tqdm.tqdm(total=len(val_loader)) as pbar:
                for i, (x, _) in enumerate(val_loader):
                    x = x.to(device)
                    model.set_forward(x)
                    pbar.update(1)
        
        # Analyze weights
        for i, module in enumerate(model.modules()):
            if isinstance(module, Attention):
                stats = module.get_weight_stats()
                if stats:
                    print(f"Attention Block {i} weight stats:")
                    if 'cosine_mean' in stats:  # 3-component format
                        print(f"  Cosine weight: {stats['cosine_mean']:.4f} ± {stats['cosine_std']:.4f}")
                        print(f"  Covariance weight: {stats['cov_mean']:.4f} ± {stats['cov_std']:.4f}")
                        print(f"  Variance weight: {stats['var_mean']:.4f} ± {stats['var_std']:.4f}")
                        print("  Distribution:")
                        for comp in ['cosine', 'cov', 'var']:
                            print(f"    {comp.capitalize()}:")
                            for bin_idx, count in enumerate(stats['histogram'][comp]):
                                bin_start = bin_idx/10
                                bin_end = (bin_idx+1)/10
                                print(f"      {bin_start:.1f}-{bin_end:.1f}: {count}")
                    else:  # Legacy format
                        print(f"  Mean: {stats['mean']:.4f}")
                        print(f"  Std: {stats['std']:.4f}")
                        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                        print("  Distribution:")
                        for bin_idx, count in enumerate(stats['histogram']):
                            bin_start = bin_idx/10
                            bin_end = (bin_idx+1)/10
                            print(f"    {bin_start:.1f}-{bin_end:.1f}: {count}")
                module.clear_weight_history()
                module.record_weights = False

    # Call this function after training
    analyze_dynamic_weights(model)

######################################################################

    print("===================================")
    print("Test phase: ")

    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()

    # Implement memory optimization
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Process batches in smaller chunks
    def direct_test(test_loader, model, params):
        acc = []
        iter_num = len(test_loader)
        
        with tqdm.tqdm(total=len(test_loader)) as pbar:
            for i, (x, _) in enumerate(test_loader):
                # Process in smaller chunks to avoid OOM
                if x.size(0) > 16:  # If batch is larger than 16
                    scores_list = []
                    chunk_size = 16
                    for j in range(0, x.size(0), chunk_size):
                        x_chunk = x[j:j+chunk_size].to(device)
                        with torch.no_grad():  # Ensure no gradients
                            scores_chunk = model.set_forward(x_chunk)
                        scores_list.append(scores_chunk.cpu())
                        torch.cuda.empty_cache()  # Clear cache after each chunk
                    scores = torch.cat(scores_list, dim=0)
                else:
                    with torch.no_grad():  # Ensure no gradients
                        x = x.to(device)
                        scores = model.set_forward(x)
                        
                pred = scores.data.cpu().numpy().argmax(axis=1)
                y = np.repeat(range(params.n_way), pred.shape[0]//params.n_way)
                acc.append(np.mean(pred == y)*100)
                pbar.set_description(
                    'Test       | Acc {:.6f}'.format(np.mean(acc)))
                pbar.update(1)
                
        acc_all = np.asarray(acc)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        return acc_mean, acc_std

######################################################################

    print("===================================")
    print("Test phase: ")

    iter_num = params.test_iter
    
    split = params.split
    if params.dataset == 'cross':
        if split == 'base':
            testfile = configs.data_dir['miniImagenet'] + 'all.json'
        else:
            testfile = configs.data_dir['CUB'] + split + '.json'
    elif params.dataset == 'cross_char':
        if split == 'base':
            testfile = configs.data_dir['Omniglot'] + 'noLatin.json'
        else:
            testfile = configs.data_dir['emnist'] + split + '.json'
    else:
        base_json = configs.data_dir[p.dataset] + "base.json"
        val_json  = configs.data_dir[p.dataset] + "val.json"

    # Data managers -------------------------------------------------
    img_sz  = 224 if "ResNet" in p.backbone else 84
    params  = dict(n_way=p.n_way, k_shot=p.k_shot, n_query=p.n_query)
    base_loader = SetDataManager(img_sz, n_episode=p.n_episode,
                                 **params).get_data_loader(base_json,aug=p.train_aug)
    val_loader  = SetDataManager(img_sz, n_episode=p.n_episode,
                                 **params).get_data_loader(val_json, aug=False)

    # Model ---------------------------------------------------------
    if "FSCT" in p.method:
        variant = "cosine" if "cosine" in p.method else "softmax"
        feat = lambda: build_feature(p.backbone, p, flatten=True)
        model = FewShotTransformer(feat, variant=variant, **params)
    elif "CTX" in p.method:
        variant = "cosine" if "cosine" in p.method else "softmax"
        feat = lambda: build_feature(p.backbone, p, flatten=False)
        model = CTX(feat, variant=variant,
                    input_dim=512 if "ResNet" in p.backbone else 64,
                    **params)
    else:
        raise ValueError("Unknown method")

    model = model.to(device)
    p.checkpoint_dir = (f"{configs.save_dir}{p.dataset}/"
                        f"{p.backbone}_{p.method}_{p.n_way}w{p.k_shot}s")
    os.makedirs(p.checkpoint_dir, exist_ok=True)

    # ----------------------------- TRAIN ---------------------------
    print("======== TRAIN ========")
    model = train(base_loader, val_loader, model,p.optimization, p.num_epoch, p)

    # ------------------------------ TEST ---------------------------
    print("======== TEST ========")
    split = p.split
    if p.dataset == "cross":
        test_json = (configs.data_dir["miniImagenet"] + "all.json"
                     if split == "base"
                     else configs.data_dir["CUB"] + f"{split}.json")
    elif p.dataset == "cross_char":
        test_json = (configs.data_dir["Omniglot"] + "noLatin.json"
                     if split == "base"
                     else configs.data_dir["emnist"] + f"{split}.json")
    else:
        test_json = configs.data_dir[p.dataset] + f"{split}.json"

    test_loader = SetDataManager(
        img_sz, n_episode=p.test_iter, **params
    ).get_data_loader(test_json, aug=False)

    # load best checkpoint
    best = get_best_file(p.checkpoint_dir)
    if best:
        model.load_state_dict(torch.load(best)["state"])

    class_names = getattr(test_loader.dataset, "class_labels", None)
    metrics = evaluate(test_loader, model, p.n_way, class_names=class_names, device=device)
    pretty_print(metrics)

    if p.wandb:
        wandb.log(metrics)
        wandb.finish()
