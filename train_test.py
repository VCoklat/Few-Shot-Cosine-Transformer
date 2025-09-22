import glob, json, os, pprint, random, time
import numpy as np, torch, tqdm
from torch.cuda.amp import GradScaler, autocast
from io_utils import parse_args, model_dict, get_best_file, get_assigned_file
from data.datamgr import SetDataManager
import configs, wandb, backbone
from methods.transformer import FewShotTransformer, Attention
from methods.CTX import CTX

# ↓ helps defragment VRAM on Kaggle
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------- #
def seed_everything(seed=4040):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def chunked_forward(model, x, chunk=16):
    if x.size(0) <= chunk:
        with torch.no_grad(): return model.set_forward(x.to(device))
    out = []
    for i in range(0, x.size(0), chunk):
        with torch.no_grad():
            out.append(model.set_forward(x[i:i+chunk].to(device)).cpu())
        torch.cuda.empty_cache()
    return torch.cat(out, 0)

# --------------------------------------------------------------------- #
# train / val
# --------------------------------------------------------------------- #
def train_loop(model, base_ldr, opt, scaler):
    model.train()
    for i,(x,_) in enumerate(base_ldr):
        opt.zero_grad(set_to_none=True)
        with autocast(): acc, loss = model.set_forward_loss(x.to(device))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if i%10==0: print(f"iter {i} | loss {loss.item():.4f} | acc {acc:.3f}")

def validate(model, val_ldr, epoch, log_wandb):
    model.eval(); acc = model.val_loop(val_ldr, epoch, log_wandb)
    return acc

def train(base_ldr, val_ldr, model, optim_name, epochs, p):
    opt = getattr(torch.optim, optim_name)(model.parameters(),
           lr=p.learning_rate, weight_decay=p.weight_decay,
           **({'momentum':p.momentum} if optim_name=="SGD" else {}))
    scaler, best = GradScaler(), 0
    for ep in range(epochs):
        train_loop(model, base_ldr, opt, scaler)
        acc = validate(model, val_ldr, ep, p.wandb)
        if acc>best:
            best=acc; torch.save({'state':model.state_dict()},
                     os.path.join(p.checkpoint_dir,'best_model.tar'))
        if ep%p.save_freq==0 or ep==epochs-1:
            torch.save({'state':model.state_dict()},
                       os.path.join(p.checkpoint_dir,f'{ep}.tar'))
    return model

# --------------------------------------------------------------------- #
# test
# --------------------------------------------------------------------- #
def direct_test(loader, model, p):
    model.eval(); res=[]
    with tqdm.tqdm(total=len(loader)) as bar:
        for x,_ in loader:
            scores = chunked_forward(model,x)
            pred=scores.numpy().argmax(1)
            y=np.repeat(range(p.n_way), len(pred)//p.n_way)
            res.append((pred==y).mean()*100)
            bar.set_description(f"Test | Acc {np.mean(res):.4f}")
            bar.update(1)
    m,s=np.mean(res),np.std(res)
    return m,s

# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #
if __name__=="__main__":
    p=parse_args(); pprint.pprint(vars(p)); print()

    # data paths -------------------------------------------------------
    if p.dataset=='cross':
        base= configs.data_dir['miniImagenet']+'all.json'
        val = configs.data_dir['CUB']+'val.json'
    elif p.dataset=='cross_char':
        base=configs.data_dir['Omniglot']+'noLatin.json'
        val =configs.data_dir['emnist']+'val.json'
    else:
        base=configs.data_dir[p.dataset]+'base.json'
        val =configs.data_dir[p.dataset]+'val.json'

    img_sz = 224 if 'ResNet' in p.backbone else 84
    fewshot = dict(n_way=p.n_way,k_shot=p.k_shot,n_query=p.n_query)
    base_mgr=SetDataManager(img_sz,n_episode=p.n_episode,**fewshot)
    val_mgr =SetDataManager(img_sz,n_episode=p.n_episode,**fewshot)
    base_ldr=base_mgr.get_data_loader(base,aug=p.train_aug)
    val_ldr =val_mgr .get_data_loader(val ,aug=False)

    # model ------------------------------------------------------------
    seed_everything()
    variant='cosine' if 'cosine' in p.method else 'softmax'
    def feat():
        return model_dict[p.backbone](p.FETI,p.dataset,flatten=True)
    model=FewShotTransformer(feat,variant=variant,**fewshot).to(device)

    # checkpoints dir --------------------------------------------------
    p.checkpoint_dir=f"{configs.save_dir}{p.dataset}/{p.backbone}_{p.method}_{p.n_way}w{p.k_shot}s"
    os.makedirs(p.checkpoint_dir,exist_ok=True)

    print("========= TRAIN ========="); model=train(base_ldr,val_ldr,model,
                       p.optimization, p.num_epoch, p)

    # --------- TEST --------------
    split=p.split
    test_json=configs.data_dir[p.dataset]+split+'.json'
    test_mgr=SetDataManager(img_sz,n_episode=p.test_iter,**fewshot)
    test_ldr=test_mgr.get_data_loader(test_json,aug=False)

    best=get_best_file(p.checkpoint_dir)
    if best: model.load_state_dict(torch.load(best)['state'])

    acc,std=direct_test(test_ldr,model,p)
    print(f"{p.test_iter} Test Acc = {acc:.2f}% ± {1.96*std/np.sqrt(p.test_iter):.2f}%")
    