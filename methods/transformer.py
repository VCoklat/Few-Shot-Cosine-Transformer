import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate
from einops import rearrange
from backbone import CosineDistLinear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---------- Helper losses ----------
def gram_squared_loss(E: torch.Tensor) -> torch.Tensor:
    """
    E : (m, d) support embeddings
    Returns scalar = Σ_{i≠j} (E_i·E_j)^2  / m²
    """
    G = E @ E.t()                       # Gram matrix (m×m)
    off_diag = G - torch.diag_embed(torch.diagonal(G))
    return off_diag.pow(2).sum() / (E.size(0) ** 2)

def cosine_distance(x1, x2):
    dots  = torch.matmul(x1, x2)
    scale = torch.einsum('bhi,bhj->bhij',
                         torch.norm(x1, 2, -1),
                         torch.norm(x2, 2, -2))
    return dots / scale

# ---------- Few-Shot Transformer ----------
class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query,
                 variant="softmax", depth=1, heads=8,
                 dim_head=64, mlp_dim=512, lambda_cov=1e-4):
        super().__init__(model_func, n_way, k_shot, n_query)

        self.loss_fn   = nn.CrossEntropyLoss()
        self.k_shot    = k_shot
        self.variant   = variant
        self.depth     = depth
        self.lambda_cov = lambda_cov   # weight for new regularizer

        dim = self.feat_dim
        self.ATTN  = Attention(dim, heads, dim_head, variant)
        self.sm    = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))

        self.FFN = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, mlp_dim),
            nn.GELU(), nn.Linear(mlp_dim, dim))

        self.linear = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim_head),
            CosineDistLinear(dim_head, 1) if variant == "cosine"
            else nn.Linear(dim_head, 1))

    # ---------- forward passes ----------
    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.view(self.n_way, self.k_shot, -1)
        z_proto   = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)
        z_query   = z_query.view(self.n_way * self.n_query, -1).unsqueeze(1)

        x, query = z_proto, z_query
        for _ in range(self.depth):
            x = self.ATTN(q=x, k=query, v=query) + x
            x = self.FFN(x) + x

        return self.linear(x).squeeze()              # (q, n)

    def set_forward_loss(self, x):
        target  = Variable(torch.from_numpy(
                   np.repeat(range(self.n_way), self.n_query)).to(device))
        scores  = self.set_forward(x)
        ce_loss = self.loss_fn(scores, target)

        # ---------- new regularizer ----------
        z_support, _ = self.parse_feature(x, is_feature=False)
        z_support = z_support.view(self.n_way * self.k_shot, -1)
        cov_loss  = gram_squared_loss(z_support)

        loss = ce_loss + self.lambda_cov * cov_loss
        acc  = (scores.argmax(1) == target).float().mean().item()

        return acc, loss

# ---------- Attention block ----------
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.8, dynamic_weight=False):
        super().__init__()
        inner_dim   = heads * dim_head
        self.heads  = heads
        self.scale  = dim_head ** -0.5
        self.variant = variant
        self.sm     = nn.Softmax(dim=-1)

        # optional learnable mixing weight (same as your original code)
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.LayerNorm(dim_head), nn.ReLU(),
                nn.Linear(dim_head, 1), nn.Sigmoid())
        else:
            self.fixed_cov_weight = nn.Parameter(
                torch.tensor(initial_cov_weight))

        self.input_linear  = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, inner_dim, bias=False))
        self.output_linear = (nn.Linear(inner_dim, dim)
                              if (heads != 1 or dim_head != dim)
                              else nn.Identity())

    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads),
            (q, k, v))

        if self.variant == "cosine":
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))

            # covariance-like component
            q_c = f_q - f_q.mean(-1, keepdim=True)
            k_c = f_k - f_k.mean(-1, keepdim=True)
            cov_component = torch.matmul(q_c, k_c.transpose(-1, -2)) / f_q.size(-1)

            if self.dynamic_weight:
                qg, kg = f_q.mean((1,2)), f_k.mean((1,2))
                w = self.weight_predictor(torch.cat([qg, kg], -1)).view(self.heads,1,1,1)
                dots = (1-w)*cosine_sim + w*cov_component
            else:
                w = torch.sigmoid(self.fixed_cov_weight)
                dots = (1-w)*cosine_sim + w*cov_component

            out = torch.matmul(dots, f_v)

        else:  # vanilla softmax attention
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out  = torch.matmul(self.sm(dots), f_v)

        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)
