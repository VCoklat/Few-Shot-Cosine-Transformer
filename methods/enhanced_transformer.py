"""
Enhanced Few-Shot Cosine Transformer with VIC regularization and Mahalanobis classifier.
Based on the architecture described in the problem statement.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat

from methods.meta_template import MetaTemplate
from methods.mahalanobis_classifier import MahalanobisClassifier
from methods.vic_regularization import VICRegularization, DynamicWeightController


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EnhancedFewShotTransformer(MetaTemplate):
    """
    Enhanced Few-Shot Cosine Transformer with:
    - Learnable weighted prototypes
    - Cosine multi-head cross-attention (2 blocks, H=4, dh=64)
    - Mahalanobis distance classifier
    - VIC regularization (Variance, Invariance, Covariance)
    - Dynamic weight controller
    """
    
    def __init__(self, model_func, n_way, k_shot, n_query, 
                 variant='cosine', depth=2, heads=4, dim_head=64, mlp_dim=512,
                 use_vic=True, use_mahalanobis=True, 
                 vic_lambda_init=None, weight_controller='uncertainty'):
        """
        Args:
            model_func: Feature extractor function
            n_way: Number of classes
            k_shot: Number of support samples per class
            n_query: Number of query samples per class
            variant: 'cosine' or 'softmax' attention
            depth: Number of transformer encoder blocks (default 2)
            heads: Number of attention heads (default 4)
            dim_head: Dimension per head (default 64)
            mlp_dim: MLP hidden dimension
            use_vic: Use VIC regularization
            use_mahalanobis: Use Mahalanobis classifier instead of cosine linear
            vic_lambda_init: Initial weights for VIC terms [λI, λV, λC]
            weight_controller: 'uncertainty' or 'gradnorm' or 'fixed'
        """
        super().__init__(model_func, n_way, k_shot, n_query)
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        self.use_vic = use_vic
        self.use_mahalanobis = use_mahalanobis
        
        dim = self.feat_dim
        
        # Learnable weighted prototypes: per-class learnable shot weights
        # Initialize to zeros so softmax gives uniform weights initially
        self.proto_weight = nn.Parameter(torch.zeros(n_way, k_shot, 1))
        self.sm = nn.Softmax(dim=1)
        
        # Cosine cross-attention blocks (2 encoder blocks)
        self.attention_blocks = nn.ModuleList([
            CosineTransformerBlock(dim, heads=heads, dim_head=dim_head, 
                                  mlp_dim=mlp_dim, variant=variant)
            for _ in range(depth)
        ])
        
        # Classifier head
        if use_mahalanobis:
            self.classifier = MahalanobisClassifier(shrinkage_alpha=None)
        else:
            # Fallback to cosine linear head
            from backbone import CosineDistLinear
            self.classifier = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                CosineDistLinear(dim_head, 1) if variant == "cosine"
                else nn.Linear(dim_head, 1)
            )
        
        # VIC regularization
        if use_vic:
            self.vic_reg = VICRegularization(target_std=1.0, eps=1e-4)
            
            # Dynamic weight controller
            if vic_lambda_init is None:
                vic_lambda_init = [9.0, 0.5, 0.5]  # [λI, λV, λC]
            
            self.weight_controller = DynamicWeightController(
                n_losses=3, method=weight_controller, 
                init_weights=vic_lambda_init, bounds=(0.25, 4.0)
            )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_prototypes(self, z_support):
        """
        Compute learnable weighted prototypes.
        
        Args:
            z_support: (n_way, k_shot, d) support embeddings
        
        Returns:
            prototypes: (n_way, d) weighted prototypes
        """
        # Apply softmax to get per-class shot weights
        weights = self.sm(self.proto_weight)  # (n_way, k_shot, 1)
        
        # Weighted sum: P[c] = Σ_i w[c,i] * z[c,i]
        prototypes = (z_support * weights).sum(dim=1)  # (n_way, d)
        
        return prototypes
    
    def apply_attention_blocks(self, prototypes, queries):
        """
        Apply cosine transformer blocks (cross-attention).
        
        Args:
            prototypes: (n_way, d) class prototypes (used as keys/values)
            queries: (n_query, d) query embeddings (used as queries)
        
        Returns:
            output: (n_query, d) transformed query embeddings
        """
        # Prepare inputs for cross-attention
        # Prototypes as keys/values, queries as queries
        P = prototypes.unsqueeze(0)  # (1, n_way, d)
        Q = queries.unsqueeze(1)  # (n_query, 1, d)
        
        # Apply transformer blocks
        H = Q  # Start with queries
        for block in self.attention_blocks:
            H = block(Q=H, K=P, V=P)
        
        # Return transformed queries
        return H.squeeze(1)  # (n_query, d)
    
    def set_forward(self, x, is_feature=False):
        """
        Forward pass for inference.
        
        Args:
            x: Input images
            is_feature: Whether input is already features
        
        Returns:
            scores: (n_query * n_way, n_way) classification scores
        """
        # Parse support and query features
        z_support, z_query = self.parse_feature(x, is_feature)
        
        # z_support: (n_way, k_shot, d)
        # z_query: (n_way, n_query, d)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        
        # Compute learnable weighted prototypes
        prototypes = self.compute_prototypes(z_support)  # (n_way, d)
        
        # Apply cosine cross-attention
        H_out = self.apply_attention_blocks(prototypes, z_query)  # (n_query * n_way, d)
        
        # Classification
        if self.use_mahalanobis:
            scores = self.classifier(H_out, z_support, prototypes)
        else:
            # Expand prototypes and compute scores
            P_expanded = prototypes.unsqueeze(0)  # (1, n_way, d)
            H_expanded = H_out.unsqueeze(1)  # (n_query * n_way, 1, d)
            scores = self.classifier(H_expanded - P_expanded)  # Basic implementation
            scores = scores.squeeze(-1)
        
        return scores
    
    def set_forward_loss(self, x):
        """
        Forward pass with loss computation for training.
        
        Args:
            x: Input images
        
        Returns:
            acc: Accuracy
            loss: Total loss (with VIC regularization if enabled)
        """
        # Parse support and query features
        z_support, z_query = self.parse_feature(x, is_feature=False)
        
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        
        # Compute learnable weighted prototypes
        prototypes = self.compute_prototypes(z_support)
        
        # Apply cosine cross-attention
        H_out = self.apply_attention_blocks(prototypes, z_query)
        
        # Classification (Invariance term)
        if self.use_mahalanobis:
            scores = self.classifier(H_out, z_support, prototypes)
        else:
            P_expanded = prototypes.unsqueeze(0)
            H_expanded = H_out.unsqueeze(1)
            scores = self.classifier(H_expanded)
            scores = scores.squeeze(-1)
        
        # Ground truth labels
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))
        
        # Classification loss (Invariance term)
        loss_I = self.loss_fn(scores, target)
        
        # Compute accuracy
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        
        # Add VIC regularization if enabled
        if self.use_vic:
            # Compute VIC losses
            loss_V, loss_C, vic_stats = self.vic_reg.compute_vic_stats(
                z_support, prototypes
            )
            
            # Dynamic weighting
            losses = [loss_I, loss_V, loss_C]
            total_loss = self.weight_controller(losses)
            
            return acc, total_loss
        else:
            return acc, loss_I


class CosineTransformerBlock(nn.Module):
    """
    Single transformer encoder block with cosine attention.
    Includes:
    - Multi-head cosine cross-attention
    - Feed-forward network with GELU
    - Layer normalization (pre-norm)
    - Residual connections
    """
    
    def __init__(self, dim, heads=4, dim_head=64, mlp_dim=512, variant='cosine'):
        super().__init__()
        self.attention = CosineMultiHeadAttention(
            dim, heads=heads, dim_head=dim_head, variant=variant
        )
        self.ffn = FeedForward(dim, mlp_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, Q, K, V):
        """
        Args:
            Q: (batch, n_q, dim) queries
            K: (batch, n_k, dim) keys
            V: (batch, n_v, dim) values
        
        Returns:
            output: (batch, n_q, dim)
        """
        # Pre-norm and attention with residual
        attn_out = self.attention(self.norm1(Q), self.norm1(K), self.norm1(V))
        Q = Q + attn_out
        
        # Pre-norm and FFN with residual
        ffn_out = self.ffn(self.norm2(Q))
        Q = Q + ffn_out
        
        return Q


class CosineMultiHeadAttention(nn.Module):
    """
    Multi-head cosine cross-attention without softmax.
    """
    
    def __init__(self, dim, heads=4, dim_head=64, variant='cosine'):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.variant = variant
        self.scale = dim_head ** -0.5
        
        # Linear projections for Q, K, V
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, Q, K, V):
        """
        Args:
            Q: (batch, n_q, dim) queries
            K: (batch, n_k, dim) keys
            V: (batch, n_v, dim) values
        
        Returns:
            output: (batch, n_q, dim)
        """
        batch = Q.shape[0]
        
        # Project and split into heads
        q = self.to_q(Q)  # (batch, n_q, inner_dim)
        k = self.to_k(K)  # (batch, n_k, inner_dim)
        v = self.to_v(V)  # (batch, n_v, inner_dim)
        
        # Reshape to separate heads: (batch, heads, n, dim_head)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        if self.variant == 'cosine':
            # Cosine similarity without softmax
            # Normalize q and k
            q_norm = F.normalize(q, p=2, dim=-1)
            k_norm = F.normalize(k, p=2, dim=-1)
            
            # Cosine attention: q @ k^T (already normalized)
            attn = torch.einsum('bhqd,bhkd->bhqk', q_norm, k_norm)
            
            # Apply to values (no softmax)
            out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        else:
            # Standard scaled dot-product attention with softmax
            attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        
        # Merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Output projection
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.
    """
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)
