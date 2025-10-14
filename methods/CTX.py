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
                input_dim = 64, dim_attn=128, gamma=0.1, epsilon=1e-8, use_regularization=True):
        super(CTX, self).__init__(model_func,  n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        self.n_way = n_way
        self.k_shot = k_shot
        self.attn = variant
        self.sm = nn.Softmax(dim=-1)
        self.dim_attn = dim_attn
        self.linear_attn = nn.Conv2d(input_dim, dim_attn, 1, bias=False)
        
        # Regularization parameters
        self.gamma = gamma  # Target value for variance (fixed to 0.1 in experiments)
        self.epsilon = epsilon  # Small scalar preventing numerical instability
        self.use_regularization = use_regularization
        
        # Dynamic weight predictor for combining three loss components
        if use_regularization:
            # Determine embedding dimension from feat_dim
            if isinstance(self.feat_dim, list):
                emb_dim = self.feat_dim[0] * self.feat_dim[1] * self.feat_dim[2]
            else:
                emb_dim = self.feat_dim
            
            # Input: global statistics from embeddings
            # Output: 3 weights (for cross-entropy, variance, covariance)
            self.weight_predictor = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),  # Concatenate support and query embeddings
                nn.LayerNorm(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 3),
                nn.Softmax(dim=-1)  # Ensure weights sum to 1
            )

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

        return euclidean_dist
    
    def variance_regularization(self, E):
        """
        Compute Variance Regularization Term (Equation 5)
        V(E) = (1/m) * sum_j max(0, gamma - sigma(E_j, epsilon))
        where sigma(E_j, epsilon) = sqrt(Var(E_j) + epsilon)
        
        Args:
            E: Embeddings tensor of shape (batch, m) where m is number of dimensions
        Returns:
            V: Variance regularization term (scalar)
        """
        # Compute variance for each dimension
        var_per_dim = torch.var(E, dim=0)  # (m,)
        
        # Compute regularized standard deviation
        sigma = torch.sqrt(var_per_dim + self.epsilon)  # (m,)
        
        # Compute hinge function
        V = torch.mean(torch.clamp(self.gamma - sigma, min=0.0))
        
        return V
    
    def covariance_regularization(self, E):
        """
        Compute Covariance Regularization Term (Equation 6)
        C(E) = (1/(m-1)) * sum_j (E_j - E_bar)(E_j - E_bar)^T
        where E_bar = (1/K) * sum_i E_j
        
        This computes the sum of squared off-diagonal coefficients of the covariance matrix.
        
        Args:
            E: Embeddings tensor of shape (batch, m) where m is number of dimensions
        Returns:
            C: Covariance regularization term (scalar)
        """
        # Center the embeddings
        E_mean = torch.mean(E, dim=0, keepdim=True)  # (1, m)
        E_centered = E - E_mean  # (batch, m)
        
        # Compute covariance matrix
        batch_size = E.size(0)
        if batch_size > 1:
            cov = torch.matmul(E_centered.T, E_centered) / (batch_size - 1)  # (m, m)
        else:
            cov = torch.matmul(E_centered.T, E_centered)  # (m, m)
        
        # Sum of squares of off-diagonal elements
        m = E.size(1)
        off_diag_mask = ~torch.eye(m, dtype=torch.bool, device=E.device)
        C = torch.sum(cov[off_diag_mask] ** 2) / m
        
        return C

    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        
        # Get embeddings for regularization computation
        z_support, z_query = self.parse_feature(x, is_feature=False)
        
        # Get prediction scores
        scores = self.set_forward(x)

        # Compute cross-entropy loss (Equation 4)
        ce_loss = self.loss_fn(scores, target)
        
        if self.use_regularization:
            # Flatten embeddings for regularization
            # E is the concatenation of support set embedding E_k and prototype embedding P_k
            z_support_flat = z_support.contiguous().view(-1, z_support.size(-1) * z_support.size(-2) * z_support.size(-3))
            z_query_flat = z_query.contiguous().view(-1, z_query.size(-1) * z_query.size(-2) * z_query.size(-3))
            E = torch.cat([z_support_flat, z_query_flat], dim=0)  # (n_way * (k_shot + n_query), d)
            
            # Compute variance regularization (Equation 5)
            var_reg = self.variance_regularization(E)
            
            # Compute covariance regularization (Equation 6)
            cov_reg = self.covariance_regularization(E)
            
            # Predict dynamic weights using global statistics
            support_global = z_support_flat.mean(dim=0)  # (d,)
            query_global = z_query_flat.mean(dim=0)  # (d,)
            global_features = torch.cat([support_global, query_global], dim=-1).unsqueeze(0)  # (1, 2*d)
            
            weights = self.weight_predictor(global_features).squeeze(0)  # (3,)
            w_ce, w_var, w_cov = weights[0], weights[1], weights[2]
            
            # Combine losses with dynamic weights
            loss = w_ce * ce_loss + w_var * var_reg + w_cov * cov_reg
        else:
            loss = ce_loss
        
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, loss
    