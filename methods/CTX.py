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
from methods.vic_regularization import VICRegularization
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from backbone import CosineDistLinear
import pdb
import IPython

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CTX(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query, heatmap=0, variant="softmax",
                input_dim = 64, dim_attn=128, use_vic=False, vic_lambda_v=1.0, 
                vic_lambda_i=1.0, vic_lambda_c=0.04, vic_dynamic_weights=True, vic_alpha=0.001):
        super(CTX, self).__init__(model_func,  n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        self.n_way = n_way
        self.k_shot = k_shot
        self.attn = variant
        self.sm = nn.Softmax(dim=-1)
        self.dim_attn = dim_attn
        self.linear_attn = nn.Conv2d(input_dim, dim_attn, 1, bias=False)
        
        # VIC Regularization
        self.use_vic = use_vic
        if self.use_vic:
            self.vic_reg = VICRegularization(
                lambda_v=vic_lambda_v,
                lambda_i=vic_lambda_i,
                lambda_c=vic_lambda_c,
                dynamic_weights=vic_dynamic_weights,
                alpha=vic_alpha
            )

    def set_forward(self, x, is_feature=False, return_embeddings=False):
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
        
        # Store original support embeddings for VIC regularization
        if return_embeddings:
            # Flatten spatial dimensions for VIC regularization
            z_support_flat = rearrange(z_support, 'n k c h w -> n k (c h w)')
        
        z_query, z_support = map(lambda t: rearrange(t, 'b n c h w -> (b n) c h w'), (z_query, z_support))

        query_q, query_v, support_k, support_v = map(lambda t: self.linear_attn(t),
                            (z_query, z_query, z_support, z_support))
        
        query_q, query_v = map(lambda t: rearrange(t, 'q c h w -> q () (c h w)'), (query_q, query_v))
        support_k, support_v = map(lambda t: rearrange(t, '(n k) c h w -> n (c h w) k',
                            n=self.n_way, k=self.k_shot), (support_k, support_v))
        
        query_q = rearrange(query_q, 'q b d -> b q d')
        
        if self.attn == 'softmax':
            dots = torch.matmul(query_q, support_k)
            scale = self.dim_attn ** 0.5
            attn_weights = self.sm(dots / scale)
            
        else:
            dots = torch.matmul(query_q, support_k)
            scale = torch.einsum('bq, nk -> nqk',
                        (torch.norm(query_q, 2, dim=-1), torch.norm(support_k, 2, dim=-2)))
            attn_weights = dots / scale
        
        out = torch.einsum('nqk, ndk -> qnd', attn_weights, support_v)
        
        euclidean_dist = -((query_v - out) ** 2).sum(dim=-1) / (self.feat_dim[1] * self.feat_dim[2])

        if return_embeddings:
            return euclidean_dist, z_support_flat
        return euclidean_dist

    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        
        if self.use_vic:
            scores, z_support = self.set_forward(x, return_embeddings=True)
        else:
            scores = self.set_forward(x)

        # Cross-entropy loss
        ce_loss = self.loss_fn(scores, target)
        
        # VIC regularization loss
        if self.use_vic and self.training:
            vic_loss, vic_dict = self.vic_reg(z_support)
            total_loss = ce_loss + vic_loss
        else:
            total_loss = ce_loss
            vic_dict = {}
        
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        
        # Store VIC loss components for logging
        self.last_vic_dict = vic_dict
        
        return acc, total_loss
    