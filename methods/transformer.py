import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from backbone import CosineDistLinear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def covariance_regularization(E):
    m = E.size(0)
    mean_E = E.mean(dim=0, keepdim=True)
    centered = E - mean_E
    cov = (centered.t() @ centered) / m
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / m
    return cov_loss

def variance_regularization(E, gamma=1.0, epsilon=1e-6):
    std_dev = torch.sqrt(torch.var(E, dim=0) + epsilon)
    zero = torch.zeros_like(std_dev)
    var_loss = torch.mean(torch.max(zero, gamma - std_dev))
    return var_loss

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
        self.sm = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim))
        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head),
            CosineDistLinear(dim_head, 1) if variant == "cosine"
            else nn.Linear(dim_head, 1))
        self.cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
        self.var_weight = nn.Parameter(torch.tensor(initial_var_weight))
    
    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)
        x, query = z_proto, z_query
        for _ in range(self.depth):
            x = self.ATTN(q=x, k=query, v=query) + x
            x = self.FFN(x) + x
        return self.linear(x).squeeze(), x.squeeze(1)  # Return scores and embeddings
    
    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))
        scores, embeddings = self.set_forward(x)
        loss = self.loss_fn(scores, target)
        cov_loss = covariance_regularization(embeddings)
        var_loss = variance_regularization(embeddings)
        total_loss = loss + torch.sigmoid(self.cov_weight) * cov_loss + torch.sigmoid(self.var_weight) * var_loss
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, total_loss

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.6, initial_var_weight=0.2, dynamic_weight=False):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim=-1)
        self.variant = variant
        
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.LayerNorm(dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, 3),
                nn.Softmax(dim=-1))
        else:
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))
        
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False))
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        self.weight_history = []
        self.record_weights = False
    
    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads), (q, k, v))
        
        if self.variant == "cosine":
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
            k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
            cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
            cov_component = cov_component / f_q.size(-1)
            q_var = torch.var(f_q, dim=-1, keepdim=True)
            k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)
            var_component = torch.matmul(q_var, k_var)
            var_component = var_component / f_q.size(-1)

            if self.dynamic_weight:
                q_global = f_q.mean(dim=(1, 2))
                k_global = f_k.mean(dim=(1, 2))
                qk_features = torch.cat([q_global, k_global], dim=-1)
                weights = self.weight_predictor(qk_features)
                if self.record_weights and not self.training:
                    self.weight_history.append(weights.detach().cpu().numpy().mean(axis=0))
                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
                dots = (cos_weight * cosine_sim +
                        cov_weight * cov_component +
                        var_weight * var_component)
            else:
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = 1.0 - cov_weight - var_weight
                dots = (cos_weight * cosine_sim +
                        cov_weight * cov_component +
                        var_weight * var_component)
            out = torch.matmul(dots, f_v)
        else:
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out = torch.matmul(self.sm(dots), f_v)

        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)

def cosine_distance(x1, x2):
    dots = torch.matmul(x1, x2)
    scale = torch.einsum('bhi, bhj -> bhij',
                        (torch.norm(x1, 2, dim=-1), torch.norm(x2, 2, dim=-2)))
    return dots / scale

def get_weight_stats(self):
    if not self.weight_history:
        return None
    weights = np.array(self.weight_history)
    if weights.shape[1] == 3:
        return {
            'cosine_mean': float(weights[:, 0].mean()),
            'cov_mean': float(weights[:, 1].mean()),
            'var_mean': float(weights[:, 2].mean()),
            'cosine_std': float(weights[:, 0].std()),
            'cov_std': float(weights[:, 1].std()),
            'var_std': float(weights[:, 2].std()),
            'histogram': {
                'cosine': np.histogram(weights[:, 0], bins=10, range=(0, 1))[0].tolist(),
                'cov': np.histogram(weights[:, 1], bins=10, range=(0, 1))[0].tolist(),
                'var': np.histogram(weights[:, 2], bins=10, range=(0, 1))[0].tolist()
            }
        }
    else:
        weights = np.array(self.weight_history)
        return {
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'min': float(weights.min()),
            'max': float(weights.max()),
            'histogram': np.histogram(weights, bins=10, range=(0, 1))[0].tolist()
        }

def clear_weight_history(self):
    self.weight_history = []
