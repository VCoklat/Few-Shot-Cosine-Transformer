"""
Few-Shot Transformer with attention-refined class prototypes
-----------------------------------------------------------
Fixes:
1. per-class prototype computation
2. correct (query, class) logits
3. device-aware target tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from einops import rearrange

from methods.meta_template import MetaTemplate
from backbone import CosineDistLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Helpers / regularisers
# ─────────────────────────────────────────────
def gram_squared_loss(E: torch.Tensor) -> torch.Tensor:
    G = E @ E.t()
    off = G - torch.diag_embed(torch.diagonal(G))
    return off.pow(2).sum() / (E.size(0) ** 2)

def variance_term(E: torch.Tensor, γ: float = 1.0, ε: float = 1e-4) -> torch.Tensor:
    flat = E.reshape(-1, E.size(-1))
    std = torch.sqrt(flat.var(dim=0) + ε)
    return torch.clamp(γ - std, min=0).mean()

def cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    dots  = torch.matmul(x1, x2)
    scale = torch.einsum("bhi,bhj->bhij",
                         torch.norm(x1, 2, -1),
                         torch.norm(x2, 2, -2))
    return dots / scale

# ─────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────
class FewShotTransformer(MetaTemplate):
    def __init__(self,
                 model_func,
                 n_way: int,
                 k_shot: int,
                 n_query: int,
                 variant: str = "softmax",
                 depth: int = 1,
                 heads: int = 8,
                 dim_head: int = 64,
                 mlp_dim: int = 512,
                 λ_cov: float = 1e-4):
        super().__init__(model_func, n_way, k_shot, n_query)
        self.k_shot, self.variant, self.depth = k_shot, variant, depth
        self.λ_cov = λ_cov

        dim = self.feat_dim
        self.ATTN = Attention(dim, heads, dim_head, variant, dynamic_weight=True)
        self.sm   = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))

        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head),
            CosineDistLinear(dim_head, 1) if variant == "cosine"
            else nn.Linear(dim_head, 1)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    # ----------------------------------------
    # Forward pass
    # ----------------------------------------
    def set_forward(self, x, is_feature=False):
        z_s, z_q = self.parse_feature(x, is_feature)          # support, query
        z_s = z_s.reshape(self.n_way, self.k_shot, -1)        # (n, k, d)
        z_q = z_q.reshape(self.n_way * self.n_query, -1)      # (n·q, d)

        # 1. weighted class prototypes  ----------------------
        proto = (z_s * self.sm(self.proto_weight)).sum(1)     # (n, d)

        # 2. attention / FFN refinement ----------------------
        proto = proto.unsqueeze(0)                            # (1, n, d)
        q     = z_q.unsqueeze(1)                              # (n·q, 1, d)

        x_ref = proto
        for _ in range(self.depth):
            x_ref = self.ATTN(q=x_ref, k=q, v=q) + x_ref
            x_ref = self.FFN(x_ref) + x_ref
        x_ref = x_ref.squeeze(0)                              # (n, d)

        # 3. pairwise logits (all queries vs all classes) ----
        q_exp     = z_q[:, None, :]                           # (n·q, 1, d)
        proto_exp = x_ref[None, :, :]                         # (1, n, d)
        logits    = -((q_exp - proto_exp) ** 2).sum(dim=2)    # (n·q, n)

        return logits                                         # (batch, n_way)

    def set_forward_loss(self, x):
        """
        Compute CE loss + covariance regulariser and
        return (episode-accuracy, total-loss).

        The target vector is now derived from the *actual*
        batch size seen by the loss, so it stays valid
        even if the number of query images per class
        changes in the dataloader.
        """
        # 1. logits for every query image
        scores = self.set_forward(x)            # (B, n_way)

        # 2. build ground-truth labels that exactly match B
        B          = scores.size(0)             # real batch
        repeats    = B // self.n_way            # queries per class
        targets    = torch.arange(self.n_way, device=scores.device) \
                        .repeat_interleave(repeats)   # (B,)

        # 3. cross-entropy + optional covariance term
        loss  = self.loss_fn(scores, targets)

        z_s, _ = self.parse_feature(x, is_feature=False)
        z_s    = z_s.reshape(self.n_way * self.k_shot, -1)
        loss  += self.λ_cov * gram_squared_loss(z_s)

        # 4. episode accuracy
        acc = (scores.argmax(1) == targets).float().mean().item()
        return acc, loss

# ─────────────────────────────────────────────
# Attention module (unchanged)
# ─────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 heads: int,
                 dim_head: int,
                 variant: str,
                 dynamic_weight: bool = True,
                 initial_mix: float = 0.9):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.scale, self.variant = heads, dim_head**-0.5, variant
        self.sm = nn.Softmax(dim=-1)
        self.dynamic = dynamic_weight

        if self.dynamic:
            self.predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.LayerNorm(dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, 1),
                nn.Sigmoid()
            )
        else:
            self.fixed_mix = nn.Parameter(torch.tensor(initial_mix))

        self.in_proj  = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, inner, bias=False))
        self.out_proj = (nn.Identity()
                         if (heads == 1 and dim_head == dim)
                         else nn.Linear(inner, dim))

        self.weight_hist, self.record = [], False

    def forward(self, q, k, v):
        f_q, f_k, f_v = map(
            lambda t: rearrange(self.in_proj(t), "q n (h d) -> h q n d", h=self.heads),
            (q, k, v)
        )

        if self.variant == "cosine":
            cos = cosine_distance(f_q, f_k.transpose(-1, -2))
            q_c = f_q - f_q.mean(-1, keepdim=True)
            k_c = f_k - f_k.mean(-1, keepdim=True)
            cov = torch.matmul(q_c, k_c.transpose(-1, -2)) / f_q.size(-1)

            if self.dynamic:
                v_pen = variance_term(f_q) + variance_term(f_k)
                qg, kg = f_q.mean((1, 2)), f_k.mean((1, 2))
                w = self.predictor(torch.cat([qg, kg], -1))   # (H, 1)
                w = w / (1.0 + v_pen)
                if self.record and not self.training:
                    self.weight_hist.append(w.mean().item())
                w = w.view(self.heads, 1, 1, 1)
                dots = (1 - w) * cos + w * cov
            else:
                mix  = torch.sigmoid(self.fixed_mix)
                dots = (1 - mix) * cos + mix * cov

            out = torch.matmul(dots, f_v)
        else:  # softmax attention
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out  = torch.matmul(self.sm(dots), f_v)

        out = rearrange(out, "h q n d -> q n (h d)")
        return self.out_proj(out)

    # Optional helpers
    def get_weight_stats(self):
        if not self.weight_hist:
            return None
        arr = np.array(self.weight_hist)
        return dict(mean=float(arr.mean()),
                    std=float(arr.std()),
                    min=float(arr.min()),
                    max=float(arr.max()))

    def clear_weight_history(self):
        self.weight_hist = []
