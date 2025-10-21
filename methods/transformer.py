import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from methods.meta_template import MetaTemplate
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from backbone import CosineDistLinear
import pdb
import IPython

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func,  n_way, k_shot, n_query, variant = "softmax",
                depth = 1, heads = 8, dim_head = 64, mlp_dim = 512):
        super(FewShotTransformer, self).__init__(model_func,  n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim

        self.ATTN = Attention(dim, heads = heads, dim_head = dim_head, variant = variant)
        
        self.sm = nn.Softmax(dim = -2)
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
        
    def set_forward(self, x, is_feature=False):

        z_support, z_query = self.parse_feature(x, is_feature)
                
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)                         # (1, n, d)
        
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)                # (q, 1, d)

        x, query = z_proto, z_query
        
        for _ in range(self.depth):
           x = self.ATTN(q = x, k = query, v = query) + x
           x = self.FFN(x) + x
        
        # Output is the probabilistic prediction for each class
        return self.linear(x).squeeze()                                                                # (q, n)
    
    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        scores = self.set_forward(x)
        
        loss = self.loss_fn(scores, target)
        predict = torch.argmax(scores, dim = 1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, loss

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim = -1)
        self.variant = variant
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias = False))
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        
        # Dynamic weight learning for variance-covariance weighting
        self.dynamic_weight = nn.Parameter(torch.ones(1))
        self.variance_weight = nn.Parameter(torch.ones(1))
        self.covariance_weight = nn.Parameter(torch.ones(1))
        
        # Invariance projection layer - uses dim_head since it operates after rearrange
        self.invariance_proj = nn.Sequential(
            nn.Linear(dim_head, dim_head),
            nn.LayerNorm(dim_head)
        )
        
    def compute_variance(self, x):
        """Compute variance across the feature dimension"""
        # x shape: (h, q, n, d)
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return variance
    
    def compute_covariance(self, x, y):
        """Compute covariance between two feature sets"""
        # x, y shape: (h, q, n, d)
        x_mean = x.mean(dim=-1, keepdim=True)
        y_mean = y.mean(dim=-1, keepdim=True)
        covariance = ((x - x_mean) * (y - y_mean)).mean(dim=-1, keepdim=True)
        return covariance
    
    def apply_invariance(self, x):
        """Apply invariance transformation for robust features"""
        # Reshape for processing
        orig_shape = x.shape
        x_flat = rearrange(x, 'h q n d -> (h q n) d')
        x_inv = self.invariance_proj(x_flat)
        x_inv = rearrange(x_inv, '(h q n) d -> h q n d', h=orig_shape[0], q=orig_shape[1], n=orig_shape[2])
        return x_inv
        
    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) ->  h q n d', h = self.heads), (q, k ,v))    
        
        # Apply invariance transformation
        f_q_inv = self.apply_invariance(f_q)
        f_k_inv = self.apply_invariance(f_k)
        
        # Compute variance and covariance statistics
        var_q = self.compute_variance(f_q)
        var_k = self.compute_variance(f_k)
        cov_qk = self.compute_covariance(f_q, f_k)
        
        # Dynamic weighting based on variance and covariance
        weight_factor = torch.sigmoid(self.dynamic_weight * (
            self.variance_weight * (var_q + var_k) + 
            self.covariance_weight * cov_qk
        ))
        
        if self.variant == "cosine":
            # Use invariance-transformed features for attention computation
            dots = cosine_distance(f_q_inv, f_k_inv.transpose(-1, -2))
            # Apply dynamic weighting
            dots = dots * weight_factor
            out = torch.matmul(dots, f_v)                                                              # (h, q, n, d_h)
        
        else: # self.variant == "softmax"
            dots = torch.matmul(f_q_inv, f_k_inv.transpose(-1, -2)) * self.scale
            # Apply dynamic weighting
            dots = dots * weight_factor
            out = torch.matmul(self.sm(dots), f_v)
        
        out = rearrange(out, 'h q n d -> q n (h d)')                                                   # (q, n, d)
        return self.output_linear(out)                                              


def cosine_distance(x1, x2):
    '''
    x1      =  [b, h, n, k]
    x2      =  [b, h, k, m]
    output  =  [b, h, n, m]
    '''
    dots = torch.matmul(x1, x2)
    scale = torch.einsum('bhi, bhj -> bhij', 
            (torch.norm(x1, 2, dim = -1), torch.norm(x2, 2, dim = -2)))
    return (dots / (scale + 1e-8))  # Add epsilon for numerical stability
