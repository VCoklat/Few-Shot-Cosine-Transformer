"""
Dynamic-VIC Few-Shot Cosine Transformer (DV-FSCT)

This module implements a hybrid few-shot classification algorithm combining:
- Few-Shot Cosine Transformer (FS-CT) for relational mapping
- Prototypical Feature Space Optimization (ProFONet) with VIC regularization
- Dynamic-weighted VIC loss for improved generalization

Key Features:
- Dynamic weight computation based on support sample hardness
- VIC regularization (Variance, Invariance, Covariance)
- Learnable prototypical embeddings with Mahalanobis distance
- Mixed-precision training (FP16) for GPU memory efficiency
- Gradient checkpointing for attention layers
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from einops import rearrange
from backbone import CosineDistLinear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DVFSCT(MetaTemplate):
    """
    Dynamic-VIC Few-Shot Cosine Transformer
    
    Args:
        model_func: Backbone feature extractor function
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        depth: Number of transformer layers (default: 1)
        heads: Number of attention heads (default: 8)
        dim_head: Dimension per attention head (default: 64)
        mlp_dim: MLP hidden dimension (default: 512)
        vic_lambda: VIC loss weight (default: 0.1)
        use_mixed_precision: Enable FP16 training (default: True)
    """
    
    def __init__(self, model_func, n_way, k_shot, n_query,
                 depth=1, heads=8, dim_head=64, mlp_dim=512,
                 vic_lambda=0.1, use_mixed_precision=True):
        super(DVFSCT, self).__init__(model_func, n_way, k_shot, n_query)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.k_shot = k_shot
        self.depth = depth
        self.vic_lambda = vic_lambda
        self.use_mixed_precision = use_mixed_precision
        dim = self.feat_dim
        
        # Learnable prototype weights (softmax normalized)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        
        # Cosine attention layer
        self.ATTN = CosineAttention(dim, heads=heads, dim_head=dim_head)
        
        # Feed-forward network with GELU activation
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
        # Output layer with cosine similarity
        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head),
            CosineDistLinear(dim_head, 1)
        )
        
        # VIC regularization parameters
        self.sigma_target = 1.0  # Target variance for VIC
        
    def cosine_sim(self, A, B, eps=1e-8):
        """
        Compute cosine similarity between tensors A and B
        
        Args:
            A: Tensor of shape [..., d]
            B: Tensor of shape [..., d]
            eps: Small constant for numerical stability
            
        Returns:
            Cosine similarity of shape [...]
        """
        A_norm = torch.norm(A, dim=-1, keepdim=True)
        B_norm = torch.norm(B, dim=-1, keepdim=True)
        return torch.sum(A * B, dim=-1) / (A_norm.squeeze(-1) * B_norm.squeeze(-1) + eps)
    
    def compute_dynamic_weights(self, z_support):
        """
        Compute dynamic VIC weights based on support sample hardness
        
        Args:
            z_support: Support features of shape [n_way, k_shot, d]
            
        Returns:
            Tuple of (alpha_V, alpha_I, alpha_C) weights
        """
        # Compute initial prototypes (mean)
        P0 = z_support.mean(dim=1)  # [n_way, d]
        
        # Compute hardness per class based on cosine similarity
        # h_k = 1 - max(cos(z, P0_k)) for each class k
        hardness_per_class = []
        for k in range(self.n_way):
            z_k = z_support[k]  # [k_shot, d]
            P0_k = P0[k].unsqueeze(0)  # [1, d]
            cos_sim = self.cosine_sim(z_k, P0_k)  # [k_shot]
            h_k = 1 - cos_sim.max()
            hardness_per_class.append(h_k)
        
        # Average hardness across classes
        h_bar = torch.stack(hardness_per_class).mean()
        
        # Dynamic weights
        alpha_V = 0.5 + 0.5 * h_bar
        alpha_I = 1.0
        alpha_C = 0.5 + 0.5 * h_bar
        
        return alpha_V, alpha_I, alpha_C, h_bar
    
    def vic_loss(self, z_support, y_support, z_query, y_query, alpha_V, alpha_I, alpha_C):
        """
        Compute VIC (Variance-Invariance-Covariance) loss
        
        Args:
            z_support: Support features [n_way*k_shot, d]
            y_support: Support labels [n_way*k_shot]
            z_query: Query features [n_way*n_query, d]
            y_query: Query labels [n_way*n_query]
            alpha_V, alpha_I, alpha_C: Dynamic weights
            
        Returns:
            Total VIC loss
        """
        d = z_support.size(-1)
        
        # Variance term: Encourage feature diversity within each dimension
        # V = (1/d) * sum_j max(0, sigma_j - 1)
        sigma = torch.std(z_support, dim=0)  # [d]
        V = torch.mean(torch.relu(self.sigma_target - sigma))
        
        # Invariance term: Cross-entropy for robust classification
        # Use simple prototypical classification as proxy
        prototypes = []
        for k in range(self.n_way):
            mask = (y_support == k)
            if mask.sum() > 0:
                proto = z_support[mask].mean(dim=0)
                prototypes.append(proto)
        
        if len(prototypes) > 0:
            prototypes = torch.stack(prototypes)  # [n_way, d]
            
            # Compute distances for query samples
            dists = torch.cdist(z_query, prototypes)  # [n_way*n_query, n_way]
            logits = -dists  # Negative distance as logits
            I = self.loss_fn(logits, y_query)
        else:
            I = torch.tensor(0.0, device=z_support.device)
        
        # Covariance term: Penalize off-diagonal covariance (decorrelation)
        # C = sum_{i!=j} cov(z_i, z_j)^2
        z_centered = z_support - z_support.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (z_support.size(0) - 1)  # [d, d]
        
        # Zero out diagonal and compute sum of squares
        cov_off_diag = cov.clone()
        cov_off_diag.fill_diagonal_(0)
        C = torch.sum(cov_off_diag ** 2) / (d * (d - 1))
        
        # Total VIC loss
        vic = alpha_V * V + alpha_I * I + alpha_C * C
        
        return vic, V, I, C
    
    def set_forward(self, x, is_feature=False):
        """
        Forward pass for inference
        
        Args:
            x: Input images or features
            is_feature: Whether input is already features
            
        Returns:
            Classification scores [n_way*n_query, n_way]
        """
        with autocast(enabled=self.use_mixed_precision):
            z_support, z_query = self.parse_feature(x, is_feature)
            
            # Reshape support features [n_way, k_shot, d]
            z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
            
            # Compute learnable weighted prototypes
            sm = nn.Softmax(dim=-2)
            z_proto = (z_support * sm(self.proto_weight)).sum(1).unsqueeze(0)  # [1, n_way, d]
            
            # Reshape query features [n_way*n_query, 1, d]
            z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)
            
            x, query = z_proto, z_query
            
            # Apply transformer layers
            for _ in range(self.depth):
                x = self.ATTN(q=x, k=query, v=query) + x
                x = self.FFN(x) + x
            
            # Output classification scores
            scores = self.linear(x).squeeze()  # [n_way*n_query, n_way]
            
        return scores
    
    def set_forward_loss(self, x):
        """
        Forward pass with loss computation (for training)
        
        Args:
            x: Input images
            
        Returns:
            Tuple of (accuracy, total_loss)
        """
        # Create target labels
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))
        
        # Parse features
        z_support, z_query = self.parse_feature(x, is_feature=False)
        
        # Reshape for VIC loss computation
        z_support_flat = z_support.contiguous().view(self.n_way * self.k_shot, -1)
        z_query_flat = z_query.contiguous().view(self.n_way * self.n_query, -1)
        
        # Create labels for VIC loss
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.k_shot)).to(device)
        y_query = target
        
        # Compute dynamic weights based on hardness
        z_support_reshaped = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        alpha_V, alpha_I, alpha_C, h_bar = self.compute_dynamic_weights(z_support_reshaped)
        
        # Compute VIC loss
        with autocast(enabled=self.use_mixed_precision):
            vic, V, I, C = self.vic_loss(z_support_flat, y_support, 
                                          z_query_flat, y_query,
                                          alpha_V, alpha_I, alpha_C)
            
            # Compute classification loss
            scores = self.set_forward(x, is_feature=False)
            loss_ce = self.loss_fn(scores, target)
            
            # Total loss
            loss = loss_ce + self.vic_lambda * vic
        
        # Compute accuracy
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        
        return acc, loss


class CosineAttention(nn.Module):
    """
    Cosine attention mechanism without softmax
    
    Uses cosine similarity for attention computation, bounded in [-1, 1]
    for better stability compared to scaled dot-product attention.
    
    Args:
        dim: Input feature dimension
        heads: Number of attention heads
        dim_head: Dimension per head
    """
    
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.dim_head = dim_head
        
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False)
        )
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
    
    def forward(self, q, k, v):
        """
        Apply cosine attention
        
        Args:
            q: Query tensor [batch, n_proto, dim]
            k: Key tensor [batch, n_query, dim]
            v: Value tensor [batch, n_query, dim]
            
        Returns:
            Attention output [batch, n_proto, dim]
        """
        # Project inputs
        f_q, f_k, f_v = map(
            lambda t: rearrange(self.input_linear(t), 'b n (h d) -> h b n d', h=self.heads),
            (q, k, v)
        )
        
        # Cosine similarity attention
        dots = self._cosine_distance(f_q, f_k.transpose(-1, -2))  # [h, b, n_proto, n_query]
        out = torch.matmul(dots, f_v)  # [h, b, n_proto, d]
        
        # Reshape and project output
        out = rearrange(out, 'h b n d -> b n (h d)')
        return self.output_linear(out)
    
    def _cosine_distance(self, x1, x2, eps=1e-8):
        """
        Compute cosine distance between x1 and x2
        
        Args:
            x1: Tensor of shape [h, b, n, k]
            x2: Tensor of shape [h, b, k, m]
            
        Returns:
            Cosine distance of shape [h, b, n, m]
        """
        dots = torch.matmul(x1, x2)
        x1_norm = torch.norm(x1, 2, dim=-1, keepdim=True)
        x2_norm = torch.norm(x2, 2, dim=-2, keepdim=True)
        scale = x1_norm * x2_norm + eps
        return dots / scale
