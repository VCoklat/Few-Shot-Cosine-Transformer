"""
Paper: "CrossTransformers: spatially-aware few-shot transfer"
Arxiv: https://arxiv.org/abs/2007.11498
This code is modified from https://github.com/lucidrains/cross-transformers-pytorch
"""


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


class CTX(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query, heatmap=0, variant="softmax",
                input_dim = 64, dim_attn=128):
        super(CTX, self).__init__(model_func,  n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        self.n_way = n_way
        self.k_shot = k_shot
        self.attn = variant
        self.sm = nn.Softmax(dim=-1)
        self.dim_attn = dim_attn
        self.linear_attn = nn.Conv2d(input_dim, dim_attn, 1, bias=False)
        
        # Dynamic weight learning for variance-covariance weighting
        self.dynamic_weight = nn.Parameter(torch.ones(1))
        self.variance_weight = nn.Parameter(torch.ones(1))
        self.covariance_weight = nn.Parameter(torch.ones(1))
        
        # Invariance projection layers
        self.invariance_query = nn.Sequential(
            nn.Linear(dim_attn, dim_attn),
            nn.LayerNorm(dim_attn)
        )
        self.invariance_support = nn.Sequential(
            nn.Linear(dim_attn, dim_attn),
            nn.LayerNorm(dim_attn)
        )

    def compute_variance(self, x):
        """Compute variance across the feature dimension"""
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return variance
    
    def compute_covariance(self, x, y):
        """Compute covariance between two feature sets"""
        x_mean = x.mean(dim=-1, keepdim=True)
        y_mean = y.mean(dim=-1, keepdim=True)
        covariance = ((x - x_mean) * (y - y_mean)).mean(dim=-1, keepdim=True)
        return covariance

    def set_forward(self, x, is_feature=False):
        """
        dimensions names:
        
        b - batch
        n - n way
        k - k shot
        q - num query
        c, h, w - img shape
        d - dim
        """

        z_support, z_query = self.parse_feature(x, is_feature)
        z_query, z_support = map(lambda t: rearrange(t, 'b n c h w -> (b n) c h w'), (z_query, z_support))

        query_q, query_v, support_k, support_v = map(lambda t: self.linear_attn(t),
                            (z_query, z_query, z_support, z_support))
        
        query_q, query_v = map(lambda t: rearrange(t, 'q c h w -> q () (c h w)'), (query_q, query_v))
        support_k, support_v = map(lambda t: rearrange(t, '(n k) c h w -> n (c h w) k',
                            n=self.n_way, k=self.k_shot), (support_k, support_v))
        
        query_q = rearrange(query_q, 'q b d -> b q d')
        
        # Apply invariance transformations
        query_q_orig = query_q
        support_k_orig = support_k
        
        query_q_flat = rearrange(query_q, 'b q d -> (b q) d')
        query_q_inv = self.invariance_query(query_q_flat)
        query_q = rearrange(query_q_inv, '(b q) d -> b q d', b=query_q_orig.shape[0])
        
        support_k_flat = rearrange(support_k, 'n d k -> (n k) d')
        support_k_inv = self.invariance_support(support_k_flat)
        support_k = rearrange(support_k_inv, '(n k) d -> n d k', n=self.n_way, k=self.k_shot)
        
        # Compute variance and covariance
        var_q = self.compute_variance(query_q)
        var_k = self.compute_variance(support_k.transpose(-1, -2))
        
        # Compute covariance between query and support
        query_q_exp = query_q.unsqueeze(0).expand(self.n_way, -1, -1, -1)
        support_k_exp = support_k.transpose(-1, -2).unsqueeze(1).expand(-1, query_q.shape[0], -1, -1)
        cov_qk = self.compute_covariance(query_q_exp, support_k_exp)
        
        # Dynamic weighting based on variance and covariance
        weight_factor = torch.sigmoid(self.dynamic_weight * (
            self.variance_weight * (var_q.mean() + var_k.mean()) + 
            self.covariance_weight * cov_qk.mean()
        ))
        
        if self.attn == 'softmax':
            dots = torch.matmul(query_q, support_k)
            scale = self.dim_attn ** 0.5
            attn_weights = self.sm(dots / scale) * weight_factor
            
        else:
            dots = torch.matmul(query_q, support_k)
            scale = torch.einsum('bq, nk -> nqk',
                        (torch.norm(query_q, 2, dim=-1), torch.norm(support_k, 2, dim=-2)))
            attn_weights = (dots / (scale + 1e-8)) * weight_factor
        
        out = torch.einsum('nqk, ndk -> qnd', attn_weights, support_v)
        
        euclidean_dist = -((query_v - out) ** 2).sum(dim=-1) / (self.feat_dim[1] * self.feat_dim[2])

        return euclidean_dist

    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        
        scores = self.set_forward(x)

        loss = self.loss_fn(scores, target)
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, loss
    