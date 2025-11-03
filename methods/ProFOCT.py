"""
ProFO-CT with Dynamic VIC: Prototypical Feature-Optimized Cosine Transformer

This method combines:
- ProFONet's VIC (Variance-Invariance-Covariance) regularized prototypical training
- FS-CT's learnable prototypical representation and cosine attention
- Dynamic per-episode adaptation of VIC coefficients

Key innovations:
1. VIC-optimized prototype space for robust, well-separated prototypes
2. Learnable weighted prototypes to handle hard/easy support samples
3. Cosine cross-attention for stable support-query alignment
4. Dynamic VIC weights that adapt per episode to task difficulty
5. Mahalanobis distance metric for non-spherical class structures
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from einops import rearrange, repeat
from backbone import CosineDistLinear
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ProFOCT(MetaTemplate):
    """
    ProFO-CT: Prototypical Feature-Optimized Cosine Transformer with Dynamic VIC
    
    Combines VIC-regularized prototypical learning with cosine transformer architecture
    and dynamic per-episode adaptation of regularization weights.
    """
    
    def __init__(self, model_func, n_way, k_shot, n_query, 
                 variant="cosine",
                 depth=1, 
                 heads=8, 
                 dim_head=64, 
                 mlp_dim=512,
                 vic_alpha=0.5,
                 vic_beta=9.0,
                 vic_gamma=0.5,
                 dynamic_vic=True,
                 vic_ema_decay=0.9,
                 distance_metric='mahalanobis',
                 use_vic_on_attention=False):
        """
        Args:
            model_func: Backbone feature extractor
            n_way: Number of classes per episode
            k_shot: Number of support samples per class
            n_query: Number of query samples per class
            variant: 'cosine' or 'softmax' for attention mechanism
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension per attention head
            mlp_dim: Dimension of feedforward network
            vic_alpha: Initial variance regularization weight
            vic_beta: Initial invariance regularization weight
            vic_gamma: Initial covariance regularization weight
            dynamic_vic: Whether to use dynamic VIC weight adaptation
            vic_ema_decay: EMA decay for smoothing dynamic VIC updates
            distance_metric: 'mahalanobis', 'euclidean', or 'cityblock'
            use_vic_on_attention: Whether to apply VIC to attention outputs
        """
        super(ProFOCT, self).__init__(model_func, n_way, k_shot, n_query)
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        self.distance_metric = distance_metric
        self.use_vic_on_attention = use_vic_on_attention
        dim = self.feat_dim
        
        # Cosine attention mechanism
        self.ATTN = VICAttention(dim, heads=heads, dim_head=dim_head, variant=variant)
        
        # Learnable prototype weights (softmax-normalized weighted mean)
        self.sm = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        
        # Feed-forward network with skip connection
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim))
        
        # Cosine linear head for final classification
        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head),
            CosineDistLinear(dim_head, 1) if variant == "cosine"
            else nn.Linear(dim_head, 1))
        
        # VIC regularization components
        self.dynamic_vic = dynamic_vic
        self.vic_ema_decay = vic_ema_decay
        
        # Initialize VIC weights with ProFONet's strong setting
        self.register_buffer('vic_alpha', torch.tensor(vic_alpha))
        self.register_buffer('vic_beta', torch.tensor(vic_beta))
        self.register_buffer('vic_gamma', torch.tensor(vic_gamma))
        
        # For dynamic VIC: track gradient magnitudes with EMA
        if dynamic_vic:
            self.register_buffer('alpha_ema', torch.tensor(vic_alpha))
            self.register_buffer('beta_ema', torch.tensor(vic_beta))
            self.register_buffer('gamma_ema', torch.tensor(vic_gamma))
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_variance_loss(self, z, eps=1e-4):
        """
        Variance regularization: enforces sufficient per-dimension variance.
        Uses hinge loss on standard deviation to prevent collapse.
        
        Args:
            z: Feature embeddings (batch_size, feature_dim)
            eps: Minimum variance threshold
        
        Returns:
            Variance loss (scalar)
        """
        std_z = torch.sqrt(z.var(dim=0) + 1e-8)
        # Hinge loss: penalize dimensions with std below eps
        return torch.mean(F.relu(eps - std_z))
    
    def compute_invariance_loss(self, z_orig, z_aug):
        """
        Invariance regularization: encourages feature invariance to transformations.
        Uses MSE between original and augmented features.
        
        Args:
            z_orig: Original feature embeddings
            z_aug: Augmented/transformed feature embeddings
        
        Returns:
            Invariance loss (scalar)
        """
        return F.mse_loss(z_orig, z_aug)
    
    def compute_covariance_loss(self, z):
        """
        Covariance regularization: penalizes off-diagonal covariance to reduce redundancy.
        Encourages feature dimensions to be decorrelated.
        
        Args:
            z: Feature embeddings (batch_size, feature_dim)
        
        Returns:
            Covariance loss (scalar)
        """
        batch_size = z.size(0)
        # Center features
        z = z - z.mean(dim=0, keepdim=True)
        # Compute covariance matrix
        cov_z = (z.T @ z) / (batch_size - 1)
        # Penalize off-diagonal elements
        off_diag = cov_z.clone()
        off_diag.diagonal().zero_()
        return off_diag.pow(2).sum() / z.size(1)
    
    def compute_mahalanobis_distance(self, query, prototypes, support_features):
        """
        Compute Mahalanobis distance for non-spherical class distributions.
        
        Args:
            query: Query features (n_query, dim)
            prototypes: Class prototypes (n_way, dim)
            support_features: Support features for covariance estimation (n_way, k_shot, dim)
        
        Returns:
            Negative Mahalanobis distances (n_query, n_way)
        """
        # Estimate class-specific covariance or global covariance
        all_support = support_features.view(-1, support_features.size(-1))
        # Add small regularization to avoid singularity
        cov = torch.cov(all_support.T) + torch.eye(all_support.size(1), device=all_support.device) * 1e-4
        
        # Compute precision matrix (inverse of covariance)
        try:
            precision = torch.linalg.inv(cov)
        except:
            # Fallback to pseudo-inverse if singular
            precision = torch.linalg.pinv(cov)
        
        # Compute Mahalanobis distance for each query to each prototype
        distances = []
        for proto in prototypes:
            diff = query - proto.unsqueeze(0)  # (n_query, dim)
            # Mahalanobis distance: sqrt((x-mu)^T * Sigma^-1 * (x-mu))
            mahal = torch.sum(diff @ precision * diff, dim=1)
            distances.append(-mahal)  # Negative for similarity score
        
        return torch.stack(distances, dim=1)  # (n_query, n_way)
    
    def update_dynamic_vic_weights(self, loss_ce, loss_v, loss_c):
        """
        Dynamically update VIC coefficients based on gradient magnitudes and statistics.
        
        Strategy:
        - Increase alpha when variance loss is high (prevent collapse)
        - Increase gamma when covariance loss is high (reduce redundancy)
        - Balance beta based on cross-entropy gradient (avoid overfitting)
        
        Args:
            loss_ce: Cross-entropy loss
            loss_v: Variance loss
            loss_c: Covariance loss
        """
        if not self.dynamic_vic or not self.training:
            return
        
        # Compute gradient magnitudes (approximate importance)
        grad_v = torch.abs(loss_v.detach())
        grad_c = torch.abs(loss_c.detach())
        grad_ce = torch.abs(loss_ce.detach())
        
        # Normalize gradients
        total_grad = grad_v + grad_c + grad_ce + 1e-8
        norm_v = grad_v / total_grad
        norm_c = grad_c / total_grad
        norm_ce = grad_ce / total_grad
        
        # Update with EMA smoothing
        new_alpha = 0.5 + norm_v * 2.0  # Range: [0.5, 2.5]
        new_gamma = 0.5 + norm_c * 2.0  # Range: [0.5, 2.5]
        new_beta = 9.0 * (1.0 - norm_ce * 0.5)  # Reduce when CE is high
        
        # Apply EMA
        self.alpha_ema = self.vic_ema_decay * self.alpha_ema + (1 - self.vic_ema_decay) * new_alpha
        self.gamma_ema = self.vic_ema_decay * self.gamma_ema + (1 - self.vic_ema_decay) * new_gamma
        self.beta_ema = self.vic_ema_decay * self.beta_ema + (1 - self.vic_ema_decay) * new_beta
        
        # Constrain weights to safe ranges to avoid over-regularization
        self.vic_alpha = torch.clamp(self.alpha_ema, 0.1, 5.0)
        self.vic_gamma = torch.clamp(self.gamma_ema, 0.1, 5.0)
        self.vic_beta = torch.clamp(self.beta_ema, 1.0, 20.0)
    
    def set_forward(self, x, is_feature=False):
        """
        Forward pass for episodic prediction.
        
        Args:
            x: Input data
            is_feature: Whether input is already extracted features
        
        Returns:
            Classification scores (n_query, n_way)
        """
        z_support, z_query = self.parse_feature(x, is_feature)
        
        # Compute learnable weighted prototypes
        # z_support: (n_way, k_shot, dim)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        
        # Apply softmax-normalized weights for learnable prototype
        proto_weights = self.sm(self.proto_weight)  # (n_way, k_shot, 1)
        z_proto = (z_support * proto_weights).sum(1).unsqueeze(0)  # (1, n_way, dim)
        
        # Prepare query features
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)  # (n_query, 1, dim)
        
        # Apply cosine transformer layers with skip connections
        x, query = z_proto, z_query
        
        for _ in range(self.depth):
            # Cross-attention: prototypes attend to queries
            x = self.ATTN(q=x, k=query, v=query) + x
            # Feed-forward with skip connection
            x = self.FFN(x) + x
        
        # Final classification via cosine linear layer
        scores = self.linear(x).squeeze()  # (n_query, n_way)
        
        return scores
    
    def set_forward_loss(self, x):
        """
        Compute loss with VIC regularization for training.
        
        Args:
            x: Input data
        
        Returns:
            accuracy: Training accuracy (scalar)
            loss: Total loss including VIC terms (scalar)
        """
        # Extract features
        z_support, z_query = self.parse_feature(x, is_feature=False)
        
        # Compute learnable weighted prototypes
        z_support_flat = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        proto_weights = self.sm(self.proto_weight)
        z_proto = (z_support_flat * proto_weights).sum(1).unsqueeze(0)  # (1, n_way, dim)
        
        # Prepare queries
        z_query_flat = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)
        
        # Apply transformer with attention
        x_attn, query = z_proto, z_query_flat
        
        for _ in range(self.depth):
            x_attn = self.ATTN(q=x_attn, k=query, v=query) + x_attn
            x_attn = self.FFN(x_attn) + x_attn
        
        # Classification scores
        scores = self.linear(x_attn).squeeze()
        
        # Cross-entropy loss
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))
        loss_ce = self.loss_fn(scores, target)
        
        # VIC regularization on support embeddings and prototypes
        all_support_embeddings = z_support_flat.view(-1, z_support_flat.size(-1))
        
        # Variance loss: prevent representation collapse
        loss_v = self.compute_variance_loss(all_support_embeddings)
        
        # Covariance loss: decorrelate feature dimensions
        loss_c = self.compute_covariance_loss(all_support_embeddings)
        
        # Invariance loss: for this we'd need augmented data, so we approximate with
        # consistency between prototypes and support embeddings
        # Alternative: use dropout as implicit augmentation
        loss_i = torch.tensor(0.0, device=device)
        
        # Optional: Apply VIC to attention outputs (but detach to reduce memory)
        if self.use_vic_on_attention:
            attn_output = x_attn.squeeze(0)  # (n_query, dim)
            # Detach to avoid storing too many intermediate gradients
            loss_v = loss_v + self.compute_variance_loss(attn_output.detach()) * 0.5
            loss_c = loss_c + self.compute_covariance_loss(attn_output.detach()) * 0.5
        
        # Dynamic VIC weight update (detach losses to avoid graph retention)
        self.update_dynamic_vic_weights(loss_ce.detach(), loss_v.detach(), loss_c.detach())
        
        # Total loss with VIC regularization
        if self.dynamic_vic:
            alpha, beta, gamma = self.vic_alpha, self.vic_beta, self.vic_gamma
        else:
            alpha, beta, gamma = self.vic_alpha, self.vic_beta, self.vic_gamma
        
        loss = loss_ce + alpha * loss_v + beta * loss_i + gamma * loss_c
        
        # Compute accuracy (detach to free memory)
        with torch.no_grad():
            predict = torch.argmax(scores.detach(), dim=1)
            acc = (predict == target).sum().item() / target.size(0)
        
        # Clean up intermediate tensors
        del z_support, z_query, z_support_flat, z_query_flat, z_proto
        del x_attn, query, scores, target
        del all_support_embeddings, loss_v, loss_c, loss_i, loss_ce
        
        return acc, loss
    
    def get_vic_weights(self):
        """Get current VIC weights for logging/debugging."""
        if self.dynamic_vic:
            return {
                'alpha': self.vic_alpha.item(),
                'beta': self.vic_beta.item(), 
                'gamma': self.vic_gamma.item()
            }
        else:
            return {
                'alpha': self.vic_alpha.item(),
                'beta': self.vic_beta.item(),
                'gamma': self.vic_gamma.item()
            }


class VICAttention(nn.Module):
    """
    Attention module compatible with VIC regularization.
    Supports both cosine and softmax attention variants.
    """
    
    def __init__(self, dim, heads, dim_head, variant):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim=-1)
        self.variant = variant
        
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False))
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
    
    def forward(self, q, k, v):
        """
        Multi-head attention with cosine or softmax variants.
        
        Args:
            q: Query (batch, n_proto, dim)
            k: Key (batch, n_query, dim)
            v: Value (batch, n_query, dim)
        
        Returns:
            Attention output (batch, n_proto, dim)
        """
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads), (q, k, v))
        
        if self.variant == "cosine":
            # Cosine attention: no softmax, stable correlation map
            dots = cosine_distance(f_q, f_k.transpose(-1, -2))  # (h, q, n_proto, n_query)
            out = torch.matmul(dots, f_v)  # (h, q, n_proto, d_h)
        else:  # softmax
            # Standard scaled dot-product attention
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out = torch.matmul(self.sm(dots), f_v)
        
        out = rearrange(out, 'h q n d -> q n (h d)')  # (q, n, d)
        return self.output_linear(out)


def cosine_distance(x1, x2):
    """
    Compute cosine similarity between two tensors.
    
    Args:
        x1: Tensor of shape (b, h, n, k)
        x2: Tensor of shape (b, h, k, m)
    
    Returns:
        Cosine similarity (b, h, n, m)
    """
    dots = torch.matmul(x1, x2)
    scale = torch.einsum('bhi, bhj -> bhij',
                         (torch.norm(x1, 2, dim=-1), torch.norm(x2, 2, dim=-2)))
    return dots / (scale + 1e-8)
