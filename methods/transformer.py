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
from torch.utils.checkpoint import checkpoint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func,  n_way, k_shot, n_query, variant = "softmax",
                depth = 1, heads = 8, dim_head = 64, mlp_dim = 512, 
                use_variance=True, use_covariance=True, use_dynamic_weights=True):
        super(FewShotTransformer, self).__init__(model_func,  n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        self.use_variance = use_variance
        self.use_covariance = use_covariance
        self.use_dynamic_weights = use_dynamic_weights
        dim = self.feat_dim

        self.ATTN = Attention(dim, heads = heads, dim_head = dim_head, variant = variant,
                             use_variance=use_variance, use_covariance=use_covariance)
        
        self.sm = nn.Softmax(dim = -2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        
        # Dynamic weight generator based on feature statistics
        if use_dynamic_weights:
            self.weight_generator = nn.Sequential(
                nn.Linear(dim * 2, dim),  # Input: concatenated mean and variance
                nn.ReLU(),
                nn.Linear(dim, k_shot),
                nn.Softmax(dim=-1)
            )
        
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
        
        # Compute dynamic weights based on variance and mean if enabled
        if self.use_dynamic_weights:
            # Compute mean and variance across shots for each way
            support_mean = z_support.mean(dim=1)  # (n_way, dim)
            support_var = z_support.var(dim=1, unbiased=False) + 1e-6  # Add epsilon for stability
            
            # Concatenate mean and variance for weight generation
            support_stats = torch.cat([support_mean, support_var], dim=-1)  # (n_way, 2*dim)
            
            # Generate dynamic weights for each way
            dynamic_weights = []
            for i in range(self.n_way):
                weights = self.weight_generator(support_stats[i])  # (k_shot,)
                dynamic_weights.append(weights.unsqueeze(-1))  # (k_shot, 1)
            dynamic_weights = torch.stack(dynamic_weights, dim=0)  # (n_way, k_shot, 1)
            
            # Combine static and dynamic weights
            combined_weights = self.sm(self.proto_weight) * dynamic_weights
            z_proto = (z_support * combined_weights).sum(1).unsqueeze(0)  # (1, n, d)
        else:
            z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
        
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)                # (q, 1, d)

        x, query = z_proto, z_query
        
        # Use gradient checkpointing to save memory
        for _ in range(self.depth):
           # Apply attention with checkpoint for memory efficiency
           x = checkpoint(self._attention_forward, x, query, use_reentrant=False) + x
           x = checkpoint(self._ffn_forward, x, use_reentrant=False) + x
        
        # Output is the probabilistic prediction for each class
        return self.linear(x).squeeze()                                                                # (q, n)
    
    def _attention_forward(self, x, query):
        """Wrapper for attention forward to use with checkpoint"""
        return self.ATTN(q=x, k=query, v=query)
    
    def _ffn_forward(self, x):
        """Wrapper for FFN forward to use with checkpoint"""
        return self.FFN(x)
    
    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        scores = self.set_forward(x)
        
        loss = self.loss_fn(scores, target)
        predict = torch.argmax(scores, dim = 1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, loss

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, use_variance=True, use_covariance=True):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim = -1)
        self.variant = variant
        self.use_variance = use_variance
        self.use_covariance = use_covariance
        
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias = False))
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        
        # Learnable parameters for variance and covariance weighting
        if use_variance:
            self.variance_scale = nn.Parameter(torch.ones(1))
        if use_covariance:
            self.covariance_scale = nn.Parameter(torch.ones(1))
            
    def compute_variance_attention(self, f_q, f_k):
        """
        Compute variance-based attention weights
        f_q: (h, q1, n1, d) - can be (h, 1, n_way, d) or (h, n_queries, n_way, d)
        f_k: (h, q2, n2, d) - typically (h, n_queries, 1, d) for queries
        Returns: variance-based weights (h, q2, n1, 1)
        """
        # Compute variance along feature dimension
        var_q = f_q.var(dim=-1, unbiased=False) + 1e-6  # (h, q1, n1)
        var_k = f_k.var(dim=-1, unbiased=False) + 1e-6  # (h, q2, n2)
        
        h, q1, n1 = var_q.shape
        h2, q2, n2 = var_k.shape
        
        # Handle different query dimension cases
        if q1 == 1 and q2 > 1:
            # Case 1: prototypes (1, n_way) vs queries (n_queries, 1)
            # var_q: (h, 1, n_way) -> need to broadcast to (h, n_queries, n_way)
            # var_k: (h, n_queries, 1) -> need to broadcast to (h, n_queries, n_way)
            var_q_expanded = var_q.expand(h, q2, n1)  # (h, n_queries, n_way)
            var_k_expanded = var_k.expand(h, q2, n1)  # (h, n_queries, n_way)
            
            var_diff = torch.abs(var_k_expanded - var_q_expanded) + 1e-6
            
        elif q1 == q2:
            # Case 2: both have same query dimension
            # var_q: (h, n_queries, n_way)
            # var_k: (h, n_queries, 1)
            var_k_expanded = var_k.expand(h, q2, n1)  # (h, n_queries, n_way)
            
            var_diff = torch.abs(var_k_expanded - var_q) + 1e-6
            
        else:
            # Fallback: use mean pooling
            var_q_mean = var_q.mean(dim=1, keepdim=True)  # (h, 1, n1)
            var_k_mean = var_k.mean(dim=2, keepdim=True)  # (h, q2, 1)
            var_q_expanded = var_q_mean.expand(h, q2, n1)
            var_k_expanded = var_k_mean.expand(h, q2, n1)
            var_diff = torch.abs(var_k_expanded - var_q_expanded) + 1e-6  # (h, q2, n1)
        
        # Variance-based similarity: inverse of variance difference
        var_weights = 1.0 / var_diff
        
        # Normalize to prevent numerical issues
        var_weights = var_weights / (var_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        return var_weights.unsqueeze(-1)  # (h, q2, n1, 1) to match with dots shape
    
    def compute_covariance_attention(self, f_q, f_k):
        """
        Compute covariance-based attention to capture feature correlations
        f_q: (h, q1, n1, d) - can be (h, 1, n_way, d) or (h, n_queries, n_way, d)
        f_k: (h, q2, n2, d) - typically (h, n_queries, 1, d) for queries
        Returns: covariance-based weights compatible with attention (h, q2, n1, 1)
        """
        # Normalize features
        f_q_norm = f_q - f_q.mean(dim=-1, keepdim=True)
        f_k_norm = f_k - f_k.mean(dim=-1, keepdim=True)
        
        # For cross-attention between prototypes and queries
        # We need to compute covariance between each query and each prototype
        
        # Handle different shapes of f_q
        h, q1, n1, d = f_q_norm.shape
        h2, q2, n2, d2 = f_k_norm.shape
        
        # Reshape for proper matrix multiplication
        # Target: compute (h, q2, n1) covariance scores
        
        if q1 == 1 and q2 > 1:
            # Case 1: prototypes (1, n_way) vs queries (n_queries, 1)
            # f_q: (h, 1, n_way, d) -> (h, n_way, d)
            f_q_reshaped = f_q_norm.squeeze(1)
            # f_k: (h, n_queries, 1, d) -> (h, n_queries, d)
            f_k_reshaped = f_k_norm.squeeze(2)
            
            # Compute: (h, n_queries, d) @ (h, d, n_way) -> (h, n_queries, n_way)
            cov = torch.matmul(f_k_reshaped, f_q_reshaped.transpose(-2, -1))
            
        elif q1 == q2:
            # Case 2: both have same query dimension (after first attention layer)
            # f_q: (h, n_queries, n_way, d)
            # f_k: (h, n_queries, 1, d)
            
            # Compute covariance for each query independently
            # Squeeze the n2=1 dimension from f_k
            f_k_reshaped = f_k_norm.squeeze(2)  # (h, n_queries, d)
            
            # For each query, compute covariance with all prototypes
            # (h, n_queries, d) @ (h, n_queries, d, n_way) -> (h, n_queries, n_way)
            # We need to compute this as batch matrix multiply
            cov = torch.einsum('hqd,hqnd->hqn', f_k_reshaped, f_q_norm)
            
        else:
            # Fallback: use mean pooling
            f_q_mean = f_q_norm.mean(dim=1)  # (h, n1, d)
            f_k_mean = f_k_norm.mean(dim=2)  # (h, q2, d)
            cov = torch.matmul(f_k_mean, f_q_mean.transpose(-2, -1))  # (h, q2, n1)
        
        cov = cov / (d + 1e-6)
        
        # Apply sigmoid for bounded output
        cov_weights = torch.sigmoid(cov)
        
        return cov_weights.unsqueeze(-1)  # (h, q2, n1, 1) to match dots shape
        
    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) ->  h q n d', h = self.heads), (q, k ,v))    
        
        if self.variant == "cosine":
            # Base cosine similarity - handle the special case where k,v might be (q, 1, d)
            # f_q: (h, q, n, d), f_k: (h, q, 1, d) typically in few-shot scenario
            # After transpose: f_k.transpose(-1, -2): (h, q, d, 1)
            dots = cosine_distance(f_q, f_k.transpose(-1, -2))  # (h, q, n, 1)
            
            # Add variance-based attention if enabled
            if self.use_variance:
                var_weights = self.compute_variance_attention(f_q, f_k)  # (h, q, n, 1)
                dots = dots + self.variance_scale * var_weights
            
            # Add covariance-based attention if enabled  
            if self.use_covariance:
                cov_weights = self.compute_covariance_attention(f_q, f_k)  # (h, q, n, 1)
                dots = dots + self.covariance_scale * cov_weights
            
            out = torch.matmul(dots, f_v)  # (h, q, n, d)
        
        else: # self.variant == "softmax"
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale  # (h, q, n, 1) or (h, q, n, n)
            
            # Add variance-based modulation
            if self.use_variance:
                var_weights = self.compute_variance_attention(f_q, f_k)  # (h, q, n, 1)
                dots = dots + self.variance_scale * var_weights
            
            # Add covariance-based modulation
            if self.use_covariance:
                cov_weights = self.compute_covariance_attention(f_q, f_k)  # (h, q, n, 1)
                dots = dots + self.covariance_scale * cov_weights
                
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
    return (dots / scale)
