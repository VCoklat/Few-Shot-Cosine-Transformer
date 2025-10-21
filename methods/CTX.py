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
        
        # Invariance projection layers - applied per channel
        self.invariance_query = nn.Sequential(
            nn.Conv2d(dim_attn, dim_attn, 1),
            nn.BatchNorm2d(dim_attn)
        )
        self.invariance_support = nn.Sequential(
            nn.Conv2d(dim_attn, dim_attn, 1),
            nn.BatchNorm2d(dim_attn)
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
        
        # Apply invariance transformations before flattening (on spatial features)
        query_q_inv = self.invariance_query(query_q)
        support_k_inv = self.invariance_support(support_k)
        
        query_q, query_v = map(lambda t: rearrange(t, 'q c h w -> q () (c h w)'), (query_q_inv, query_v))
        support_k, support_v = map(lambda t: rearrange(t, '(n k) c h w -> n (c h w) k',
                            n=self.n_way, k=self.k_shot), (support_k_inv, support_v))
        
        query_q = rearrange(query_q, 'q b d -> b q d')
        
        # Compute variance for dynamic weighting
        var_q = self.compute_variance(query_q).mean()
        # Reshape support_k for variance computation: (n, d, k) -> (n*k, d)
        support_k_reshaped = support_k.transpose(-1, -2).reshape(-1, support_k.shape[1])
        var_k = self.compute_variance(support_k_reshaped).mean()
        
        # Compute covariance metric using feature statistics
        # Simple approximation: correlation of norms
        query_norm = torch.norm(query_q, p=2, dim=-1).mean()
        support_norm = torch.norm(support_k, p=2, dim=1).mean()
        cov_qk = query_norm * support_norm
        
        # Dynamic weighting based on variance and covariance
        weight_factor = torch.sigmoid(self.dynamic_weight * (
            self.variance_weight * (var_q + var_k) + 
            self.covariance_weight * cov_qk
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
    