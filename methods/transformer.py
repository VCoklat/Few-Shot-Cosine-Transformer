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
    def __init__(self, model_func, n_way, k_shot, n_query, variant="softmax",
                depth=1, heads=8, dim_head=64, mlp_dim=512,
                initial_cov_weight=0.3, initial_var_weight=0.5, dynamic_weight=False):
        super(FewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim

        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                             initial_cov_weight=initial_cov_weight,
                             initial_var_weight=initial_var_weight,
                             dynamic_weight=dynamic_weight)
        
        self.sm = nn.Softmax(dim = -2)
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

        Targets are now built from the *actual* batch size,
        so the loss stays consistent even if the number of
        query images per class changes.
        """
        # 1. logits for every query image
        scores = self.set_forward(x)            # (B, n_way)

        # 2. derive ground-truth labels that match B
        B = scores.size(0)                      # real batch

        if B == 0:                              # safety guard
            raise ValueError(
                "Empty batch encountered in set_forward_loss"
            )

        repeats = B // self.n_way               # queries per class
        targets = torch.arange(self.n_way, device=scores.device) \
                     .repeat_interleave(repeats)       # (B,)

        # 3. cross-entropy + optional covariance term
        loss = self.loss_fn(scores, targets)

        z_s, _ = self.parse_feature(x, is_feature=False)
        z_s = z_s.reshape(self.n_way * self.k_shot, -1)
        loss += self.λ_cov * gram_squared_loss(z_s)

        # 4. episode accuracy
        acc = (scores.argmax(1) == targets).float().mean().item()
        return acc, loss


# ─────────────────────────────────────────────
# Attention module (unchanged)
# ─────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.6, initial_var_weight=0.2, dynamic_weight=False):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim = -1)
        self.variant = variant
        
        # Dynamic weighting components
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            # Network to predict the weights based on features (now 3 components)
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.LayerNorm(dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, 3),  # Now predict 3 weights instead of 1
                nn.Softmax(dim=-1)  # Ensure weights sum to 1.0
            )
        else:
            # Fixed weights as parameters (still learnable)
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))
            
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias = False))
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        
        self.weight_history = []  # To store weights for analysis
        self.record_weights = False  # Toggle for weight recording
    
    def forward(self, q, k, v):
        f_q, f_k, f_v = map(
            lambda t: rearrange(self.in_proj(t), "q n (h d) -> h q n d", h=self.heads),
            (q, k, v)
        )

        if self.variant == "cosine":
            # Calculate cosine similarity (invariance component)
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            
            # Calculate covariance component
            q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
            k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
            cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
            cov_component = cov_component / f_q.size(-1)
            
            # Calculate variance component (new)
            # Compute variance along feature dimension
            q_var = torch.var(f_q, dim=-1, keepdim=True)  # [h, q, n, 1]
            k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)  # [h, q, 1, m]
            
            # Create variance-based attention
            var_component = torch.matmul(q_var, k_var)  # [h, q, n, m]
            var_component = var_component / f_q.size(-1)  # Scale like covariance
            
            if self.dynamic_weight:
                # Use global feature statistics
                q_global = f_q.mean(dim=(1, 2))  # [h, d]
                k_global = f_k.mean(dim=(1, 2))  # [h, d]
                
                # Concatenate global query and key features
                qk_features = torch.cat([q_global, k_global], dim=-1)  # [h, 2d]
                
                # Predict three weights per attention head
                weights = self.weight_predictor(qk_features)  # [h, 3]
                
                # Record weights during evaluation if needed
                if self.record_weights and not self.training:
                    self.weight_history.append(weights.detach().cpu().numpy().mean(axis=0))
                
                # Extract individual weights
                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)  # Cosine weight
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)  # Covariance weight
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)  # Variance weight
                
                # Combine all three components
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
            else:
                # Use fixed weights
                cov_weight = torch.sigmoid(self.fixed_cov_weight) 
                var_weight = torch.sigmoid(self.fixed_var_weight)
                # Ensure weights sum to approximately 1 by using the remaining portion for cosine
                cos_weight = 1.0 - cov_weight - var_weight
                
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
                
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
        
        weights = np.array(self.weight_history)
        if weights.shape[1] == 3:  # We have 3 components
            return {
                'cosine_mean': float(weights[:, 0].mean()),
                'cov_mean': float(weights[:, 1].mean()),
                'var_mean': float(weights[:, 2].mean()),
                'cosine_std': float(weights[:, 0].std()),
                'cov_std': float(weights[:, 1].std()),
                'var_std': float(weights[:, 2].std()),
                'histogram': {
                    'cosine': np.histogram(weights[:, 0], bins=10, range=(0,1))[0].tolist(),
                    'cov': np.histogram(weights[:, 1], bins=10, range=(0,1))[0].tolist(),
                    'var': np.histogram(weights[:, 2], bins=10, range=(0,1))[0].tolist()
                }
            }
        else:  # Legacy format with single weight
            weights = np.array(self.weight_history)
            return {
                'mean': float(weights.mean()),
                'std': float(weights.std()),
                'min': float(weights.min()),
                'max': float(weights.max()),
                'histogram': np.histogram(weights, bins=10, range=(0,1))[0].tolist()
            }
    
    def clear_weight_history(self):
        self.weight_hist = []
