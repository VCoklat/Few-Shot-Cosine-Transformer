"""
Enhanced Few-Shot Cosine Transformer with VIC Regularization and Mahalanobis Classifier
Based on the problem statement architecture combining:
- Cosine cross-attention (from FS-CT paper)
- VIC regularization (from VICReg and ProFONet)
- Mahalanobis distance classifier (from ProFONet and Mahalanobis-FSL)
- Dynamic weighting controller
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from methods.vic_regularization import VICRegularization, MahalanobisClassifier, DynamicWeightController
from einops import rearrange
from backbone import CosineDistLinear
import warnings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EnhancedFewShotTransformer(MetaTemplate):
    """
    Enhanced Few-Shot Transformer with:
    - Learnable weighted prototypes
    - Cosine multi-head cross-attention (2 encoder blocks)
    - Mahalanobis distance classifier
    - VIC regularization with dynamic weighting
    """
    def __init__(self, model_func, n_way, k_shot, n_query, 
                 variant="cosine", depth=2, heads=4, dim_head=64, mlp_dim=512,
                 use_vic=True, vic_weight_strategy='uncertainty',
                 shrinkage_param=None, enable_amp=False):
        super(EnhancedFewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        self.use_vic = use_vic
        self.enable_amp = enable_amp
        
        dim = self.feat_dim
        
        # Learnable weighted prototypes (per-class, per-shot weights)
        self.proto_weight = nn.Parameter(torch.zeros(n_way, k_shot, 1))
        self.sm = nn.Softmax(dim=-2)
        
        # Cosine cross-attention blocks (2 blocks as per specification)
        self.attention_blocks = nn.ModuleList([
            CosineAttentionBlock(dim, heads=heads, dim_head=dim_head, 
                               mlp_dim=mlp_dim, variant=variant)
            for _ in range(depth)
        ])
        
        # Mahalanobis classifier
        if shrinkage_param is None:
            shrinkage_param = dim / (k_shot + dim) if k_shot > 0 else 0.1
        self.mahalanobis_classifier = MahalanobisClassifier(shrinkage_param=shrinkage_param)
        
        # VIC regularization
        if use_vic:
            self.vic_regularizer = VICRegularization(feature_dim=dim)
            self.weight_controller = DynamicWeightController(
                strategy=vic_weight_strategy,
                initial_weights={'invariance': 9.0, 'variance': 0.5, 'covariance': 0.5}
            )
        
        # Classification loss
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_weighted_prototypes(self, z_support):
        """
        Compute learnable weighted prototypes
        
        Args:
            z_support: (n_way, k_shot, d)
        Returns:
            prototypes: (n_way, d)
        """
        # Apply softmax to get normalized weights
        weights = self.sm(self.proto_weight)  # (n_way, k_shot, 1)
        
        # Weighted sum: z_proto[c] = Σ_i w[c,i] * z_support[c,i]
        z_proto = (z_support * weights).sum(dim=1)  # (n_way, d)
        
        return z_proto
    
    def cosine_cross_attention(self, prototypes, queries):
        """
        Apply cosine cross-attention between prototypes and queries
        
        Args:
            prototypes: (n_way, d) - class prototypes
            queries: (n_query, d) - query embeddings
        Returns:
            output: (n_query, d) - attended query features
        """
        # Reshape for attention: prototypes as keys/values, queries as queries
        # In cross-attention, prototypes attend to queries
        x = prototypes.unsqueeze(0)  # (1, n_way, d)
        query = queries.unsqueeze(1)  # (n_query, 1, d)
        
        # Apply attention blocks
        for block in self.attention_blocks:
            x = block(q=x, k=query, v=query)
        
        # Output shape: (1, n_way, d) or similar
        # We want per-query embeddings, so we need to transpose/reshape
        # Actually, we want to attend queries to prototypes for classification
        # Let me fix this: queries should attend to prototypes
        
        # Correct approach: queries are Q, prototypes are K and V
        x = queries.unsqueeze(1)  # (n_query, 1, d)
        kv = prototypes.unsqueeze(0)  # (1, n_way, d)
        
        for block in self.attention_blocks:
            x = block(q=x, k=kv, v=kv)
        
        # Output: (n_query, 1, d) -> squeeze -> (n_query, d)
        output = x.squeeze(1)
        
        return output
    
    def set_forward(self, x, is_feature=False):
        """
        Forward pass with Mahalanobis classification
        
        Args:
            x: input data
            is_feature: whether input is already features
        Returns:
            scores: (n_query, n_way) logits
        """
        # Parse features
        z_support, z_query = self.parse_feature(x, is_feature)
        
        # z_support: (n_way, k_shot, d)
        # z_query: (n_way, n_query, d)
        z_support = z_support.contiguous()
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)  # (n_way * n_query, d)
        
        # Compute learnable weighted prototypes
        z_proto = self.compute_weighted_prototypes(z_support)  # (n_way, d)
        
        # Apply cosine cross-attention
        h_out = self.cosine_cross_attention(z_proto, z_query)  # (n_query, d)
        
        # Mahalanobis classification
        # Note: We need to use the original z_support (before attention) for covariance
        scores = self.mahalanobis_classifier(
            queries=h_out,
            support_embeddings=z_support,
            prototypes=z_proto
        )
        
        return scores
    
    def set_forward_loss(self, x):
        """
        Forward pass with VIC regularization
        
        Args:
            x: input data
        Returns:
            acc: accuracy
            loss: total loss
        """
        # Parse features
        z_support, z_query = self.parse_feature(x, is_feature=False)
        
        # z_support: (n_way, k_shot, d)
        # z_query: (n_way, n_query, d)
        z_support = z_support.contiguous()
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        
        # Compute prototypes
        z_proto = self.compute_weighted_prototypes(z_support)
        
        # VIC regularization on support embeddings and prototypes
        if self.use_vic:
            # Flatten support embeddings for VIC
            z_support_flat = z_support.view(-1, z_support.shape[-1])
            
            # Compute VIC losses
            loss_V, loss_C = self.vic_regularizer(z_support_flat, z_proto)
        else:
            loss_V = torch.tensor(0.0, device=z_support.device)
            loss_C = torch.tensor(0.0, device=z_support.device)
        
        # Apply cosine cross-attention
        h_out = self.cosine_cross_attention(z_proto, z_query)
        
        # Mahalanobis classification (Invariance term)
        scores = self.mahalanobis_classifier(
            queries=h_out,
            support_embeddings=z_support,
            prototypes=z_proto
        )
        
        # Ground truth labels
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))
        
        # Classification loss (Invariance)
        loss_I = self.loss_fn(scores, target)
        
        # Combine losses with dynamic weighting
        if self.use_vic:
            total_loss, loss_dict = self.weight_controller.compute_weighted_loss(
                loss_I, loss_V, loss_C
            )
        else:
            total_loss = loss_I
            loss_dict = {'loss_I': loss_I.item()}
        
        # Compute accuracy
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        
        # Store loss components for logging (optional)
        if hasattr(self, 'loss_components'):
            self.loss_components = loss_dict
        
        return acc, total_loss


class CosineAttentionBlock(nn.Module):
    """
    Single cosine attention block with residual connections and FFN
    """
    def __init__(self, dim, heads=4, dim_head=64, mlp_dim=512, variant='cosine'):
        super().__init__()
        
        self.attention = CosineAttention(dim, heads=heads, dim_head=dim_head, variant=variant)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network with GELU
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, q, k, v):
        """
        Args:
            q: queries (batch, n_q, dim)
            k: keys (batch, n_k, dim)
            v: values (batch, n_k, dim)
        Returns:
            output: (batch, n_q, dim)
        """
        # Pre-norm + attention + residual
        normed_q = self.norm1(q)
        attn_out = self.attention(normed_q, k, v)
        x = attn_out + q
        
        # Pre-norm + FFN + residual
        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x)
        x = ffn_out + x
        
        return x


class CosineAttention(nn.Module):
    """
    Cosine attention without softmax
    """
    def __init__(self, dim, heads=4, dim_head=64, variant='cosine'):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.variant = variant
        self.scale = dim_head ** -0.5
        
        # Linear projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v):
        """
        Args:
            q: queries (batch, n_q, dim)
            k: keys (batch, n_k, dim)
            v: values (batch, n_k, dim)
        Returns:
            output: (batch, n_q, dim)
        """
        # Project to Q, K, V
        Q = self.to_q(q)  # (batch, n_q, inner_dim)
        K = self.to_k(k)  # (batch, n_k, inner_dim)
        V = self.to_v(v)  # (batch, n_k, inner_dim)
        
        # Reshape for multi-head: (batch, n, heads, dim_head)
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.heads)
        
        if self.variant == 'cosine':
            # Cosine similarity without softmax
            # Normalize Q and K
            Q_norm = F.normalize(Q, p=2, dim=-1)
            K_norm = F.normalize(K, p=2, dim=-1)
            
            # Cosine similarity: (batch, heads, n_q, n_k)
            attn = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        else:
            # Scaled dot-product attention with softmax
            attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn = self.sm(attn)
        
        # Apply attention to values: (batch, heads, n_q, dim_head)
        out = torch.matmul(attn, V)
        
        # Reshape back: (batch, n_q, inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Output projection
        out = self.to_out(out)
        
        return out
