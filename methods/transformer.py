import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate
from einops import rearrange
from backbone import CosineDistLinear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------ #
# 1. Regularizers
# ------------------------------------------------------------------ #
def gram_squared_loss(E: torch.Tensor) -> torch.Tensor:
    G = E @ E.t()
    off = G - torch.diag_embed(torch.diagonal(G))
    return off.pow(2).sum() / (E.size(0) ** 2)

def variance_term(E: torch.Tensor, γ: float = 1.0, ε: float = 1e-4) -> torch.Tensor:
    flat = E.reshape(-1, E.size(-1))
    std  = torch.sqrt(flat.var(dim=0) + ε)
    return torch.clamp(γ - std, min=0).mean()

def cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    dots  = torch.matmul(x1, x2)
    scale = torch.einsum("bhi,bhj->bhij",
                         torch.norm(x1, 2, -1),
                         torch.norm(x2, 2, -2))
    return dots / scale

# ------------------------------------------------------------------ #
# 2. Few-Shot Transformer
# ------------------------------------------------------------------ #
class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query,
                 variant="softmax", depth=1, heads=8,
                 dim_head=64, mlp_dim=512, λ_cov=1e-4):
        super().__init__(model_func, n_way, k_shot, n_query)

        self.k_shot, self.variant, self.depth = k_shot, variant, depth
        self.λ_cov = λ_cov
        dim = self.feat_dim

        self.ATTN = Attention(dim, heads, dim_head, variant,
                              dynamic_weight=True)   # turn on dynamic branch
        self.sm = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))

        self.FFN = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, mlp_dim),
            nn.GELU(), nn.Linear(mlp_dim, dim))

        self.linear = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim_head),
            CosineDistLinear(dim_head, 1) if variant=="cosine"
            else nn.Linear(dim_head, 1))

        self.loss_fn = nn.CrossEntropyLoss()

    # -------------------------------------------------------------- #
    def set_forward(self, x, is_feature=False):
        z_s, z_q = self.parse_feature(x, is_feature)
        z_s = z_s.reshape(self.n_way, self.k_shot, -1)
        proto = (z_s * self.sm(self.proto_weight)).sum(1).unsqueeze(0)

        z_q = z_q.reshape(self.n_way * self.n_query, -1).unsqueeze(1)
        x, q = proto, z_q

        for _ in range(self.depth):
            x = self.ATTN(q=x, k=q, v=q) + x
            x = self.FFN(x) + x

        return self.linear(x).squeeze()           # (q, n)

    def set_forward_loss(self, x):
        tgt = Variable(torch.from_numpy(
              np.repeat(range(self.n_way), self.n_query)).to(device))
        scores = self.set_forward(x)
        loss   = self.loss_fn(scores, tgt)

        z_s, _ = self.parse_feature(x, is_feature=False)
        z_s = z_s.reshape(self.n_way * self.k_shot, -1)
        loss += self.λ_cov * gram_squared_loss(z_s)

        acc = (scores.argmax(1)==tgt).float().mean().item()
        return acc, loss

# ------------------------------------------------------------------ #
# 3. Attention with dynamic covariance + variance weighting
# ------------------------------------------------------------------ #
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant,
                 dynamic_weight=True, initial_mix=0.9):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.scale, self.variant = heads, dim_head**-0.5, variant
        self.sm = nn.Softmax(dim=-1)

        self.dynamic = dynamic_weight
        if self.dynamic:
            self.predictor = nn.Sequential(
                nn.Linear(dim_head*2, dim_head),
                nn.LayerNorm(dim_head), nn.ReLU(),
                nn.Linear(dim_head, 1), nn.Sigmoid())
        else:
            self.fixed_mix = nn.Parameter(torch.tensor(initial_mix))

        self.in_proj  = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, inner, bias=False))
        self.out_proj = nn.Identity() if (heads==1 and dim_head==dim) \
                        else nn.Linear(inner, dim)

        self.weight_hist, self.record = [], False

    # -------------------------------------------------------------- #
    def forward(self, q,k,v):
        f_q,f_k,f_v = map(lambda t:
            rearrange(self.in_proj(t), 'q n (h d)->h q n d', h=self.heads),
            (q,k,v))

        if self.variant=="cosine":
            cos = cosine_distance(f_q, f_k.transpose(-1,-2))

            q_c = f_q - f_q.mean(-1,keepdim=True)
            k_c = f_k - f_k.mean(-1,keepdim=True)
            cov = torch.matmul(q_c, k_c.transpose(-1,-2)) / f_q.size(-1)

            # -------- dynamic branch with variance penalty ----------
            if self.dynamic:
                v_pen = variance_term(f_q) + variance_term(f_k)
                qg, kg = f_q.mean((1,2)), f_k.mean((1,2))
                w = self.predictor(torch.cat([qg,kg],-1))      # (H,1)
                w = w / (1.0 + v_pen)                          # penalise low-var
                if self.record and not self.training:
                    self.weight_hist.append(w.mean().item())
                w = w.view(self.heads,1,1,1)
                dots = (1-w)*cos + w*cov
            else:
                mix = torch.sigmoid(self.fixed_mix)
                dots = (1-mix)*cos + mix*cov

            out = torch.matmul(dots, f_v)
        else:  # softmax
            dots = torch.matmul(f_q, f_k.transpose(-1,-2)) * self.scale
            out  = torch.matmul(self.sm(dots), f_v)

        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.out_proj(out)

    # optional logging helpers
    def get_weight_stats(self):
        if not self.weight_hist: return None
        arr = np.array(self.weight_hist)
        return dict(mean=float(arr.mean()), std=float(arr.std()),
                    min=float(arr.min()), max=float(arr.max()))

    def clear_weight_history(self): self.weight_hist=[]