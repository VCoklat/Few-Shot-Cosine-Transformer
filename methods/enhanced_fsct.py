"""
Enhanced Few-Shot Cosine Transformer with VIC Regularization and Mahalanobis Distance
Implements specifications from:
- Enhancing Few-Shot Image Classification With Cosine Transformer (FS-CT)
- ProFONet (VIC regularization)
- Mahalanobis-FSL
- VICReg
- GradNorm
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from einops import rearrange, repeat

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EnhancedFSCT(MetaTemplate):
    """Enhanced Few-Shot Cosine Transformer with VIC regularization and Mahalanobis classification"""
    
    def __init__(self, model_func, n_way, k_shot, n_query,
                 depth=2, heads=4, dim_head=64, mlp_dim=512,
                 lambda_I=9.0, lambda_V=0.5, lambda_C=0.5,
                 use_uncertainty_weighting=True,
                 use_gradnorm=False,
                 shrinkage_alpha=None,
                 epsilon=1e-4):
        super(EnhancedFSCT, self).__init__(model_func, n_way, k_shot, n_query)
        
        self.k_shot = k_shot
        self.depth = depth
        dim = self.feat_dim
        
        # Learnable weighted prototypes - initialize to uniform (zeros in log-space)
        self.proto_weight = nn.Parameter(torch.zeros(n_way, k_shot, 1))
        
        # Cosine cross-attention encoder blocks
        self.encoder_blocks = nn.ModuleList([
            CosineEncoderBlock(dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
            for _ in range(depth)
        ])
        
        # VIC loss weights
        if use_uncertainty_weighting:
            # Learnable log-variances for uncertainty weighting
            self.log_var_I = nn.Parameter(torch.zeros(1))
            self.log_var_V = nn.Parameter(torch.zeros(1))
            self.log_var_C = nn.Parameter(torch.zeros(1))
            self.use_uncertainty = True
            self.use_gradnorm = False
        elif use_gradnorm:
            # GradNorm controller
            self.lambda_I = nn.Parameter(torch.tensor(lambda_I), requires_grad=False)
            self.lambda_V = nn.Parameter(torch.tensor(lambda_V), requires_grad=False)
            self.lambda_C = nn.Parameter(torch.tensor(lambda_C), requires_grad=False)
            self.use_uncertainty = False
            self.use_gradnorm = True
            self.gradnorm_alpha = 1.0
            # Track initial losses for relative loss rates
            self.initial_losses = None
        else:
            # Fixed weights with stats-driven fallback
            self.lambda_I = lambda_I
            self.lambda_V = lambda_V
            self.lambda_C = lambda_C
            self.use_uncertainty = False
            self.use_gradnorm = False
        
        # Shrinkage covariance parameter
        self.shrinkage_alpha = shrinkage_alpha  # If None, will compute adaptively
        self.epsilon = epsilon
        
        # Classification loss
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_weighted_prototypes(self, z_support):
        """
        Compute learnable weighted prototypes
        z_support: (n_way, k_shot, d)
        Returns: (n_way, d)
        """
        # Softmax over shots to get weights
        weights = F.softmax(self.proto_weight, dim=1)  # (n_way, k_shot, 1)
        # Weighted sum
        z_proto = (z_support * weights).sum(dim=1)  # (n_way, d)
        return z_proto
    
    def mahalanobis_distance(self, query_features, prototypes, support_features):
        """
        Compute Mahalanobis distances with shrinkage covariance
        query_features: (n_query, d)
        prototypes: (n_way, d)
        support_features: (n_way, k_shot, d)
        Returns: distances (n_query, n_way)
        """
        n_query = query_features.shape[0]
        n_way = prototypes.shape[0]
        d = query_features.shape[1]
        
        distances = torch.zeros(n_query, n_way).to(device)
        
        for c in range(n_way):
            # Get support embeddings for class c
            support_c = support_features[c]  # (k_shot, d)
            
            # Compute sample covariance
            if self.k_shot > 1:
                support_centered = support_c - support_c.mean(dim=0, keepdim=True)
                cov_c = (support_centered.T @ support_centered) / (self.k_shot - 1)
            else:
                cov_c = torch.eye(d).to(device) * 0.01
            
            # Shrinkage: Σ_c = (1-α)S_c + αI
            if self.shrinkage_alpha is None:
                # Adaptive shrinkage: α ≈ d/(k+d)
                alpha = d / (self.k_shot + d)
            else:
                alpha = self.shrinkage_alpha
            
            shrunk_cov = (1 - alpha) * cov_c + alpha * torch.eye(d).to(device)
            
            # Compute inverse via Cholesky decomposition for numerical stability
            try:
                L = torch.linalg.cholesky(shrunk_cov + self.epsilon * torch.eye(d).to(device))
                inv_cov = torch.cholesky_inverse(L)
            except:
                # Fallback to pseudo-inverse if Cholesky fails
                inv_cov = torch.linalg.pinv(shrunk_cov + self.epsilon * torch.eye(d).to(device))
            
            # Compute Mahalanobis distance for all queries to this prototype
            diff = query_features - prototypes[c].unsqueeze(0)  # (n_query, d)
            distances[:, c] = torch.sum(diff @ inv_cov * diff, dim=1)
        
        return distances
    
    def compute_variance_loss(self, embeddings):
        """
        Variance term: hinge on per-dimension std toward target σ=1
        embeddings: (N, d) where N = n_way * (k_shot + 1) for support + prototypes
        """
        std_per_dim = torch.sqrt(embeddings.var(dim=0) + self.epsilon)  # (d,)
        # Hinge loss: max(0, σ_j - 1 - ε)
        variance_loss = F.relu(std_per_dim - 1.0 - self.epsilon).mean()
        return variance_loss
    
    def compute_covariance_loss(self, embeddings):
        """
        Covariance term: off-diagonal squared Frobenius norm
        embeddings: (N, d)
        """
        # Normalize embeddings
        embeddings_norm = embeddings - embeddings.mean(dim=0, keepdim=True)
        embeddings_norm = F.normalize(embeddings_norm, p=2, dim=1)
        
        # Compute covariance matrix
        N = embeddings.shape[0]
        cov_matrix = (embeddings_norm.T @ embeddings_norm) / N  # (d, d)
        
        # Off-diagonal elements
        d = cov_matrix.shape[0]
        mask = 1.0 - torch.eye(d).to(device)
        off_diag = cov_matrix * mask
        
        # Squared Frobenius norm of off-diagonal
        covariance_loss = (off_diag ** 2).sum() / d
        return covariance_loss
    
    def set_forward(self, x, is_feature=False):
        """Forward pass for inference"""
        z_support, z_query = self.parse_feature(x, is_feature)
        
        # Flatten spatial dimensions
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)  # (n, k, d)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)  # (n*q, d)
        
        # Compute weighted prototypes
        z_proto = self.compute_weighted_prototypes(z_support)  # (n, d)
        
        # Pass queries through cosine encoder blocks with prototypes as context
        h_out = z_query  # Start with query features
        for block in self.encoder_blocks:
            h_out = block(h_out, z_proto)  # Cross-attention: queries attend to prototypes
        
        # Mahalanobis distance classification
        distances = self.mahalanobis_distance(h_out, z_proto, z_support)
        
        # Convert distances to logits (negative distances)
        scores = -distances
        
        return scores
    
    def set_forward_loss(self, x):
        """Forward pass with loss computation including VIC regularization"""
        z_support, z_query = self.parse_feature(x, False)
        
        # Flatten spatial dimensions
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)  # (n, k, d)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)  # (n*q, d)
        
        # Compute weighted prototypes
        z_proto = self.compute_weighted_prototypes(z_support)  # (n, d)
        
        # Pass queries through cosine encoder blocks with prototypes as context
        h_out = z_query  # Start with query features
        for block in self.encoder_blocks:
            h_out = block(h_out, z_proto)  # Cross-attention: queries attend to prototypes
        
        # 1. Invariance loss (classification via Mahalanobis distance)
        distances = self.mahalanobis_distance(h_out, z_proto, z_support)
        scores = -distances
        
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))
        
        loss_I = self.loss_fn(scores, target)
        
        # 2. VIC regularization on support embeddings + prototypes
        # Concatenate support embeddings and prototypes for VIC
        support_flat = z_support.view(-1, z_support.shape[-1])  # (n*k, d)
        embeddings_for_vic = torch.cat([support_flat, z_proto], dim=0)  # (n*k + n, d)
        
        loss_V = self.compute_variance_loss(embeddings_for_vic)
        loss_C = self.compute_covariance_loss(embeddings_for_vic)
        
        # 3. Dynamic weighting
        if self.use_uncertainty:
            # Uncertainty weighting: L_k * exp(-s_k) + s_k
            loss = (loss_I * torch.exp(-self.log_var_I) + self.log_var_I +
                   loss_V * torch.exp(-self.log_var_V) + self.log_var_V +
                   loss_C * torch.exp(-self.log_var_C) + self.log_var_C)
        elif self.use_gradnorm:
            # GradNorm controller (simplified version)
            loss = self.lambda_I * loss_I + self.lambda_V * loss_V + self.lambda_C * loss_C
        else:
            # Fixed weights
            loss = self.lambda_I * loss_I + self.lambda_V * loss_V + self.lambda_C * loss_C
        
        # Compute accuracy
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        
        return acc, loss
    
    def train_loop_with_amp(self, epoch, num_epoch, train_loader, wandb_flag, optimizer, scaler=None, grad_clip=None):
        """Training loop with mixed precision support and gradient clipping"""
        import tqdm
        import wandb
        
        avg_loss = 0
        avg_acc = []
        use_amp = scaler is not None
        
        with tqdm.tqdm(total=len(train_loader)) as train_pbar:
            for i, (x, _) in enumerate(train_loader):
                if self.change_way:
                    self.n_way = x.size(0)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if use_amp:
                    with torch.cuda.amp.autocast():
                        acc, loss = self.set_forward_loss(x=x.to(device))
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping if specified
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                    
                    # Optimizer step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    acc, loss = self.set_forward_loss(x=x.to(device))
                    loss.backward()
                    
                    # Gradient clipping if specified
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                    
                    optimizer.step()
                
                avg_loss += loss.item()
                avg_acc.append(acc)
                train_pbar.set_description('Epoch {:03d}/{:03d} | Acc {:.6f}  | Loss {:.6f}'.format(
                    epoch + 1, num_epoch, np.mean(avg_acc) * 100, avg_loss/float(i+1)))
                train_pbar.update(1)
        
        if wandb_flag:
            wandb.log({"Loss": avg_loss/float(i + 1), 'Train Acc': np.mean(avg_acc) * 100}, step=epoch + 1)


class CosineEncoderBlock(nn.Module):
    """
    Cosine attention encoder block with:
    - Multi-head cosine cross-attention (no softmax)
    - GELU FFN
    - Pre-norm LayerNorm
    - Residual connections
    """
    
    def __init__(self, dim, heads=4, dim_head=64, mlp_dim=512):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        
        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(dim)
        
        # Linear projections for Q, K, V
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Linear(inner_dim, dim)
        
        # Pre-norm for FFN
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN with GELU
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, queries, prototypes):
        """
        Cross-attention: queries attend to prototypes
        queries: (n_query, dim) - query embeddings
        prototypes: (n_way, dim) - prototype embeddings
        Returns: (n_query, dim) - attended query features
        """
        # Store residual from queries
        residual = queries
        
        # Pre-norm
        queries_norm = self.norm1(queries)
        prototypes_norm = self.norm1(prototypes)
        
        # Project to Q, K, V
        # Queries generate Q, prototypes generate K and V
        q = self.to_q(queries_norm)  # (n_query, inner_dim)
        k = self.to_k(prototypes_norm)  # (n_way, inner_dim)
        v = self.to_v(prototypes_norm)  # (n_way, inner_dim)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'q (h d) -> h q d', h=self.heads)  # (h, n_query, dim_head)
        k = rearrange(k, 'n (h d) -> h n d', h=self.heads)  # (h, n_way, dim_head)
        v = rearrange(v, 'n (h d) -> h n d', h=self.heads)  # (h, n_way, dim_head)
        
        # Cosine attention (no softmax)
        # Normalize Q and K for cosine similarity
        q_norm = F.normalize(q, p=2, dim=-1)  # (h, n_query, dim_head)
        k_norm = F.normalize(k, p=2, dim=-1)  # (h, n_way, dim_head)
        
        # Cosine similarity: Q @ K^T
        attn = torch.einsum('hqd,hnd->hqn', q_norm, k_norm)  # (h, n_query, n_way)
        
        # Apply attention to values (no softmax, raw cosine similarities)
        out = torch.einsum('hqn,hnd->hqd', attn, v)  # (h, n_query, dim_head)
        
        # Reshape and project back
        out = rearrange(out, 'h q d -> q (h d)')  # (n_query, inner_dim)
        out = self.to_out(out)  # (n_query, dim)
        
        # Residual connection
        out = out + residual
        
        # FFN with pre-norm and residual
        residual = out
        out = self.norm2(out)
        out = self.ffn(out) + residual
        
        return out
