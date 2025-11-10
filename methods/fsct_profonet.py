"""
Hybrid FS-CT + ProFONet Algorithm for Few-Shot Classification

This module implements a hybrid approach combining:
1. Learnable Prototypical Embedding (from FS-CT) with VIC Regularization (from ProFONet)
2. Cosine Attention Transformer (from FS-CT)
3. Dynamic Weight VIC for memory-efficient training

Paper references:
- FS-CT: "Enhancing Few-shot Image Classification with Cosine Transformer"
- ProFONet: VIC Regularization for few-shot learning
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from einops import rearrange, repeat
from backbone import CosineDistLinear
import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class VICRegularization(nn.Module):
    """
    VIC Regularization Module implementing:
    - Variance Regularization (prevents norm collapse)
    - Invariance Regularization (cross-entropy loss)
    - Covariance Regularization (prevents representation collapse)
    """
    def __init__(self, gamma=1.0, epsilon=1e-6):
        super(VICRegularization, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
    
    def variance_loss(self, embeddings):
        """
        Variance Regularization: prevents norm collapse
        V(E) = (1/m) * Σ max(0, γ - σ(E_j, ε))
        where σ(E_j, ε) = sqrt(Var(E_j) + ε)
        """
        # embeddings shape: (batch_size, dim)
        std = torch.sqrt(embeddings.var(dim=0) + self.epsilon)
        variance_loss = torch.mean(F.relu(self.gamma - std))
        return variance_loss
    
    def covariance_loss(self, embeddings):
        """
        Covariance Regularization: prevents representation collapse
        C(E) = (1/(m-1)) * Σ (E_j - Ē)(E_j - Ē)^T
        C_loss = Σ(off_diagonal(C(E))^2) / m
        """
        batch_size, dim = embeddings.shape
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov_matrix = (embeddings.T @ embeddings) / (batch_size - 1)
        
        # Sum of squared off-diagonal elements
        # Create mask for off-diagonal elements
        mask = ~torch.eye(dim, dtype=torch.bool, device=embeddings.device)
        off_diagonal_loss = (cov_matrix[mask] ** 2).sum() / dim
        
        return off_diagonal_loss
    
    def forward(self, embeddings):
        """
        Compute VIC regularization losses
        Returns: (variance_loss, covariance_loss)
        """
        v_loss = self.variance_loss(embeddings)
        c_loss = self.covariance_loss(embeddings)
        return v_loss, c_loss


class DynamicWeightScheduler:
    """
    Dynamic Weight VIC Scheduler
    Adjusts regularization weights based on training progress
    """
    def __init__(self, lambda_V_base=0.5, lambda_I=9.0, lambda_C_base=0.5):
        self.lambda_V_base = lambda_V_base
        self.lambda_I = lambda_I
        self.lambda_C_base = lambda_C_base
    
    def get_weights(self, current_epoch, total_epochs):
        """
        Compute dynamic weights based on training progress
        
        λ_V = 0.5 * (1 + 0.3 * epoch_ratio)  # Increase variance weight
        λ_I = 9.0  # Keep invariance dominant
        λ_C = 0.5 * (1 - 0.2 * epoch_ratio)  # Decrease covariance weight
        """
        epoch_ratio = current_epoch / max(total_epochs, 1)
        
        lambda_V = self.lambda_V_base * (1 + 0.3 * epoch_ratio)
        lambda_I = self.lambda_I
        lambda_C = self.lambda_C_base * (1 - 0.2 * epoch_ratio)
        
        return lambda_V, lambda_I, lambda_C


class CosineAttentionLayer(nn.Module):
    """
    Cosine Attention mechanism without softmax
    Uses cosine similarity for attention computation
    """
    def __init__(self, dim, heads=4, dim_head=160, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)
        
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False)
        )
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
    
    def cosine_similarity_attention(self, q, k, v):
        """
        Compute cosine similarity-based attention (NO softmax)
        
        A = qk / (||q||_2 ⊗ ||k||_2)
        A ∈ [-1, 1], no softmax needed
        h_a = A ⊙ v
        """
        # q: (h, q, n, d_h)
        # k: (h, q, 1, d_h)
        # v: (h, q, 1, d_h)
        
        # Compute dot product
        qk = torch.matmul(q, k.transpose(-2, -1))  # (h, q, n, 1)
        
        # Compute magnitude normalization
        q_norm = torch.norm(q, p=2, dim=-1, keepdim=True)  # (h, q, n, 1)
        k_norm = torch.norm(k, p=2, dim=-1, keepdim=True).transpose(-2, -1)  # (h, q, 1, 1)
        
        # Cosine similarity
        scale = q_norm * k_norm + 1e-8
        attn_weights = qk / scale  # (h, q, n, 1)
        
        # Element-wise multiplication with values
        out = attn_weights * v  # (h, q, n, d_h)
        
        return out
    
    def forward(self, q, k, v):
        """
        Forward pass with cosine attention
        q: (q, n, d) - query features (prototypes)
        k: (q, 1, d) - key features (query samples)
        v: (q, 1, d) - value features (query samples)
        """
        # Project and reshape for multi-head attention
        f_q = rearrange(self.input_linear(q), 'q n (h d) -> h q n d', h=self.heads)
        f_k = rearrange(self.input_linear(k), 'q n (h d) -> h q n d', h=self.heads)
        f_v = rearrange(self.input_linear(v), 'q n (h d) -> h q n d', h=self.heads)
        
        # Apply cosine attention
        out = self.cosine_similarity_attention(f_q, f_k, f_v)
        
        # Reshape and project output
        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)


class FSCT_ProFONet(MetaTemplate):
    """
    Hybrid FS-CT + ProFONet Few-Shot Classifier
    
    Combines:
    1. Learnable Weighted Prototypical Embedding with VIC Regularization
    2. Cosine Attention Transformer
    3. Dynamic Weight VIC adjustment
    4. Memory-efficient training support
    """
    def __init__(self, model_func, n_way, k_shot, n_query, 
                 variant="cosine",
                 depth=1, 
                 heads=4, 
                 dim_head=160, 
                 mlp_dim=512,
                 dropout=0.0,
                 lambda_V_base=0.5,
                 lambda_I=9.0,
                 lambda_C_base=0.5,
                 gradient_checkpointing=False,
                 mixed_precision=False):
        super(FSCT_ProFONet, self).__init__(model_func, n_way, k_shot, n_query)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        dim = self.feat_dim
        
        # Learnable prototype weights
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        self.sm = nn.Softmax(dim=-2)
        
        # VIC Regularization Module
        self.vic_reg = VICRegularization(gamma=1.0, epsilon=1e-6)
        
        # Dynamic Weight Scheduler
        self.weight_scheduler = DynamicWeightScheduler(
            lambda_V_base=lambda_V_base,
            lambda_I=lambda_I,
            lambda_C_base=lambda_C_base
        )
        
        # Cosine Attention Transformer
        self.ATTN = CosineAttentionLayer(
            dim=dim, 
            heads=heads, 
            dim_head=dim_head, 
            dropout=dropout
        )
        
        # Feed-Forward Network
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
        # Cosine Linear Classification Layer
        if variant == "cosine":
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                CosineDistLinear(dim_head, 1)
            )
        else:
            self.linear = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head),
                nn.Linear(dim_head, 1)
            )
        
        # Track current epoch for dynamic weighting
        self.current_epoch = 0
        self.total_epochs = 50  # Default, will be updated during training
    
    def set_epoch(self, epoch, total_epochs):
        """Update current epoch for dynamic weight adjustment"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def compute_vic_loss(self, z_support, z_proto):
        """
        Compute VIC regularization losses
        
        Args:
            z_support: Support set embeddings (n, k, d)
            z_proto: Prototype embeddings (1, n, d) or (n, d)
        
        Returns:
            v_loss: Variance loss
            c_loss: Covariance loss
        """
        # Concatenate support embeddings and prototypes
        # E = concat([ZS.reshape(n*k, d), ZP])
        n, k, d = z_support.shape
        support_flat = z_support.reshape(n * k, d)
        
        # Ensure z_proto is 2D (n, d)
        if z_proto.dim() == 3:
            z_proto = z_proto.squeeze(0)  # (n, d)
        
        embeddings = torch.cat([support_flat, z_proto], dim=0)  # (n*k + n, d)
        
        # Compute VIC losses
        v_loss, c_loss = self.vic_reg(embeddings)
        
        return v_loss, c_loss
    
    def set_forward(self, x, is_feature=False):
        """
        Forward pass for classification
        
        1. Extract features and compute prototypes
        2. Apply cosine transformer
        3. Classify with cosine linear layer
        """
        # Extract features
        z_support, z_query = self.parse_feature(x, is_feature)
        
        # z_support: (n, k, d)
        # z_query: (q, d)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        
        # Learnable Weighted Prototypical Embedding
        # ZP = Σ(ZS ⊙ W_avg, axis=k)
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
        
        # Reshape query for attention
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)
        
        x, query = z_proto, z_query
        
        # Apply Cosine Transformer layers with skip connections
        for _ in range(self.depth):
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
                x = torch.utils.checkpoint.checkpoint(self.ATTN, x, query, query) + x
                x = torch.utils.checkpoint.checkpoint(self.FFN, x) + x
            else:
                x = self.ATTN(q=x, k=query, v=query) + x
                x = self.FFN(x) + x
        
        # Cosine Linear Classification
        scores = self.linear(x).squeeze()  # (q, n)
        
        return scores, z_support, z_proto
    
    def set_forward_loss(self, x):
        """
        Forward pass with combined loss computation
        
        L_total = λ_V * V(E) + λ_I * I + λ_C * C(E)
        """
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))
        
        # Forward pass
        scores, z_support, z_proto = self.set_forward(x)
        
        # Invariance loss (cross-entropy)
        invariance_loss = self.loss_fn(scores, target)
        
        # VIC regularization losses
        variance_loss, covariance_loss = self.compute_vic_loss(z_support, z_proto)
        
        # Get dynamic weights
        lambda_V, lambda_I, lambda_C = self.weight_scheduler.get_weights(
            self.current_epoch, self.total_epochs
        )
        
        # Combined loss
        total_loss = (lambda_V * variance_loss + 
                     lambda_I * invariance_loss + 
                     lambda_C * covariance_loss)
        
        # Compute accuracy
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        
        return acc, total_loss
    
    def correct(self, x):
        """Override correct method to handle the new forward signature"""
        scores, _, _ = self.set_forward(x)
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.to(device))
        
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels
        top1_correct = (topk_ind[:, 0] == y_query).sum().item()
        return float(top1_correct), len(y_query)
    
    def train_loop(self, epoch, num_epoch, train_loader, wandb_flag, optimizer):
        """
        Override train_loop to add:
        - Gradient clipping
        - Epoch setting for dynamic weight adjustment
        - Mixed precision training support (optional)
        """
        import tqdm
        
        # Set current epoch for dynamic weight adjustment
        self.set_epoch(epoch, num_epoch)
        
        avg_loss = 0
        avg_acc = []
        
        # Mixed precision scaler (optional)
        if self.mixed_precision:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
        
        with tqdm.tqdm(total=len(train_loader)) as train_pbar:
            for i, (x, _) in enumerate(train_loader):
                if self.change_way:
                    self.n_way = x.size(0)
                
                optimizer.zero_grad()
                
                # Forward pass with optional mixed precision
                if self.mixed_precision and torch.cuda.is_available():
                    with autocast():
                        acc, loss = self.set_forward_loss(x=x.to(device))
                    scaler.scale(loss).backward()
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    acc, loss = self.set_forward_loss(x=x.to(device))
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                
                avg_loss += loss.item()
                avg_acc.append(acc)
                train_pbar.set_description('Epoch {:03d}/{:03d} | Acc {:.6f}  | Loss {:.6f}'.format(
                    epoch + 1, num_epoch, np.mean(avg_acc) * 100, avg_loss/float(i+1)))
                train_pbar.update(1)
        
        if wandb_flag:
            import wandb
            # Log dynamic weights for monitoring
            lambda_V, lambda_I, lambda_C = self.weight_scheduler.get_weights(epoch, num_epoch)
            wandb.log({
                "Loss": avg_loss/float(i + 1),
                'Train Acc': np.mean(avg_acc) * 100,
                'lambda_V': lambda_V,
                'lambda_I': lambda_I,
                'lambda_C': lambda_C
            }, step=epoch + 1)
