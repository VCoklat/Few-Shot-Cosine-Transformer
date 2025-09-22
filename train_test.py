import os, random, time, json, pprint
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

# --------------------------------------------\---------------------- #
def change_model(name: str) -> str:
    """Swap vanilla Conv backbones for their ‘NP’ (no-pool) variants."""
    mapping = {
        "Conv4":  "Conv4NP",
        "Conv6":  "Conv6NP",
        "Conv4S": "Conv4SNP",
        "Conv6S": "Conv6SNP",
    }
    return mapping.get(name, name)

def seed_everything(seed=4040):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ------------------------------------------------------------------ #
def train(base_loader, val_loader, model, opt_name, epochs, p):
    opt = getattr(torch.optim, opt_name)(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay, **({'momentum':p.momentum} if opt_name=='SGD' else {}))
    scaler, best = GradScaler(), 0

    for ep in range(epochs):
        model.train()

        for i, (x, _) in enumerate(base_loader):
            opt.zero_grad(set_to_none=True)
            with autocast():
                acc, loss = model.set_forward_loss(x.to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if i % 10 == 0:
                print(f"ep{ep+1}/{epochs} it{i} loss {loss.item():.3f} acc {acc:.3f}")

        model.eval()
        with torch.no_grad():
            acc = model.val_loop(val_loader, ep, p.wandb)

        if acc > best:
            best = acc
            checkpoint_file = os.path.join(p.checkpoint_dir, 'best_model.tar')
            torch.save({'state': model.state_dict()}, checkpoint_file)

        if (ep % p.save_freq == 0) or (ep == epochs - 1):
            checkpoint_file = os.path.join(p.checkpoint_dir, f'{ep}.tar')
            torch.save({'state': model.state_dict()}, checkpoint_file)

        print()

    return model

# ------------------------------------------------------------------ #
def build_feature(backbone_key, p, flatten=True):
    if p.dataset in ["Omniglot","cross_char"]:
        backbone_key = change_model(backbone_key)
    if "ResNet" in backbone_key:
        return model_dict[backbone_key](p.FETI, p.dataset, flatten=flatten)
    return model_dict[backbone_key](p.dataset, flatten=flatten)

# ------------------------------------------------------------------ #
if __name__=="__main__":
    p = parse_args()
    pprint.pprint(vars(p))
    print()
    seed_everything()

    if p.wandb:
        wname = f"{p.method}_{p.backbone}_{p.dataset}_{p.n_way}w{p.k_shot}s"
        if p.train_aug: wname += "_aug"
        if p.FETI and 'ResNet' in p.backbone: wname += "_FETI"
        wname += "_" + p.datetime
        wandb.init(project="Few-Shot_TransFormer", name=wname, config=p, id=p.datetime)

    if p.dataset == "cross":
        base_json = configs.data_dir["miniImagenet"] + "all.json"
        val_json = configs.data_dir["CUB"] + "val.json"
    elif p.dataset == "cross_char":
        base_json = configs.data_dir["Omniglot"] + "noLatin.json"
        val_json = configs.data_dir["emnist"] + "val.json"
    else:
        base_json = configs.data_dir[p.dataset] + "base.json"
        val_json = configs.data_dir[p.dataset] + "val.json"

    img_sz = 224 if "ResNet" in p.backbone else 84
    params = dict(n_way=p.n_way, k_shot=p.k_shot, n_query=p.n_query)

    base_mgr = SetDataManager(img_sz, n_episode=p.n_episode, **params)
    val_mgr = SetDataManager(img_sz, n_episode=p.n_episode, **params)
    base_loader = base_mgr.get_data_loader(base_json, aug=p.train_aug)
    val_loader = val_mgr.get_data_loader(val_json, aug=False)

    if "FSCT" in p.method:
        variant = "cosine" if "cosine" in p.method else "softmax"
        feat = lambda: build_feature(p.backbone, p, flatten=True)
        model = FewShotTransformer(feat, variant=variant, **params)
    elif "CTX" in p.method:
        variant = "cosine" if "cosine" in p.method else "softmax"
        feat = lambda: build_feature(p.backbone, p, flatten=False)
        model = CTX(feat, variant=variant, input_dim=512 if "ResNet" in p.backbone else 64, **params)
    else:
        raise ValueError("Unknown method")

    model = model.to(device)

    p.checkpoint_dir = f"{configs.save_dir}{p.dataset}/{p.backbone}_{p.method}_{p.n_way}w{p.k_shot}s"
    os.makedirs(p.checkpoint_dir, exist_ok=True)

    print("======== TRAIN ========")
    model = train(base_loader, val_loader, model, p.optimization, p.num_epoch, p)

    print("======== TEST ========")

    split = p.split
    if p.dataset == "cross":
        test_json = configs.data_dir["miniImagenet"] + "all.json" if split == "base" else configs.data_dir["CUB"] + split + ".json"
    elif p.dataset == "cross_char":
        test_json = configs.data_dir["Omniglot"] + "noLatin.json" if split == "base" else configs.data_dir["emnist"] + split + ".json"
    else:
        test_json = configs.data_dir[p.dataset] + split + ".json"

    test_mgr = SetDataManager(img_sz, n_episode=p.test_iter, **params)
    test_loader = test_mgr.get_data_loader(test_json, aug=False)

    best = get_best_file(p.checkpoint_dir)
    if best:
        model.load_state_dict(torch.load(best)["state"])

    class_names = getattr(test_loader.dataset, "class_labels", None)

    metrics = evaluate(test_loader, model, p.n_way, class_names=class_names, device=device)
    pretty_print(metrics)

    if p.wandb:
        wandb.log(metrics)
        wandb.finish()