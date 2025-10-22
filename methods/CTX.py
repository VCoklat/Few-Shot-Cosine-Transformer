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
from torch.utils.checkpoint import checkpoint
import pdb
import IPython

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CTX(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query, heatmap=0, variant="softmax",
                input_dim = 64, dim_attn=128, use_variance=True, use_invariance=True):
        super(CTX, self).__init__(model_func,  n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        self.n_way = n_way
        self.k_shot = k_shot
        self.attn = variant
        self.sm = nn.Softmax(dim=-1)
        self.dim_attn = dim_attn
        self.use_variance = use_variance
        self.use_invariance = use_invariance
        
        self.linear_attn = nn.Conv2d(input_dim, dim_attn, 1, bias=False)
        
        # Variance and invariance normalization layers
        if use_variance:
            self.variance_scale = nn.Parameter(torch.ones(1))
        if use_invariance:
            # Instance normalization for translation invariance
            self.invariance_norm = nn.InstanceNorm2d(dim_attn, affine=True)

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

        # Apply attention transformation with memory-efficient checkpointing
        query_q, query_v, support_k, support_v = map(lambda t: self.linear_attn(t),
                            (z_query, z_query, z_support, z_support))
        
        # Apply invariance normalization if enabled
        if self.use_invariance:
            query_q = self.invariance_norm(query_q)
            query_v = self.invariance_norm(query_v)
            support_k = self.invariance_norm(support_k)
            support_v = self.invariance_norm(support_v)
        
        query_q, query_v = map(lambda t: rearrange(t, 'q c h w -> q () (c h w)'), (query_q, query_v))
        support_k, support_v = map(lambda t: rearrange(t, '(n k) c h w -> n (c h w) k',
                            n=self.n_way, k=self.k_shot), (support_k, support_v))
        
        query_q = rearrange(query_q, 'q b d -> b q d')
        
        if self.attn == 'softmax':
            dots = torch.matmul(query_q, support_k)
            scale = self.dim_attn ** 0.5
            
            # Add variance-based regularization if enabled
            if self.use_variance:
                # Compute variance of attention scores
                attn_var = dots.var(dim=-1, keepdim=True, unbiased=False) + 1e-6
                # Scale attention by inverse variance for stability
                variance_factor = 1.0 / (1.0 + self.variance_scale * attn_var)
                dots = dots * variance_factor
                
            attn_weights = self.sm(dots / scale)
            
        else:  # cosine variant
            dots = torch.matmul(query_q, support_k)
            
            # Compute normalization with numerical stability
            query_norm = torch.norm(query_q, 2, dim=-1).clamp(min=1e-6)  # (b, q)
            support_norm = torch.norm(support_k, 2, dim=-2).clamp(min=1e-6)  # (n, k)
            
            # Create scale with correct broadcasting: (b, q) x (n, k) -> (n, q, k)
            # We need to broadcast to match dots shape
            scale = torch.einsum('bq, nk -> nqk',
                        query_norm, support_norm)
            
            # Add variance-based modulation
            if self.use_variance:
                attn_var = dots.var(dim=-1, keepdim=True, unbiased=False) + 1e-6
                variance_factor = 1.0 / (1.0 + self.variance_scale * attn_var)
                dots = dots * variance_factor
                
            attn_weights = dots / scale
        
        out = torch.einsum('nqk, ndk -> qnd', attn_weights, support_v)
        
        # Use safer distance computation to prevent dimension mismatches
        diff = query_v - out
        # Ensure dimensions match before computing distance
        if diff.shape[-1] != (self.feat_dim[1] * self.feat_dim[2]):
            # Adapt the divisor to actual feature dimensions
            feat_size = diff.shape[-1]
        else:
            feat_size = self.feat_dim[1] * self.feat_dim[2]
            
        euclidean_dist = -(diff ** 2).sum(dim=-1) / feat_size

        return euclidean_dist

    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        
        scores = self.set_forward(x)

        loss = self.loss_fn(scores, target)
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, loss
    