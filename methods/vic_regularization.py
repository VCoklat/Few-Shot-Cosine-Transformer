"""
VIC Regularization: Variance-Invariance-Covariance regularization
Based on VICReg and ProFONet papers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegularization(nn.Module):
    """
    Implements VIC (Variance-Invariance-Covariance) regularization for few-shot learning.
    
    - Variance term V: hinge on per-dimension std toward target σ=1
    - Invariance term I: classification loss (cross-entropy)
    - Covariance term C: off-diagonal squared Frobenius norm to decorrelate features
    """
    
    def __init__(self, target_std=1.0, eps=1e-4):
        """
        Args:
            target_std: Target standard deviation for variance regularization
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.target_std = target_std
        self.eps = eps
    
    def variance_loss(self, embeddings):
        """
        Variance term: hinge on per-dimension std toward target.
        V = (1/d) Σ_j max(0, σ_j - target_std - ε)
        where σ_j = sqrt(Var(E[:, j]) + ε)
        
        Args:
            embeddings: (n, d) embeddings tensor
        
        Returns:
            variance_loss: scalar
        """
        # Compute standard deviation per dimension
        std = torch.sqrt(embeddings.var(dim=0) + self.eps)  # (d,)
        
        # Hinge loss: penalize when std is below (target - eps)
        # We want std to be at least target_std - eps
        hinge = torch.clamp(self.target_std - std - self.eps, min=0.0)
        
        # Average over dimensions
        variance_loss = hinge.mean()
        
        return variance_loss
    
    def covariance_loss(self, embeddings):
        """
        Covariance term: off-diagonal squared Frobenius norm.
        C = (1/d) Σ_{i≠j} C_ij²
        
        Encourages decorrelation between different feature dimensions.
        
        Args:
            embeddings: (n, d) embeddings tensor
        
        Returns:
            covariance_loss: scalar
        """
        n, d = embeddings.shape
        
        # Normalize embeddings
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        embeddings = embeddings / (embeddings.std(dim=0, keepdim=True) + self.eps)
        
        # Compute covariance matrix
        cov = (embeddings.T @ embeddings) / (n - 1 + self.eps)  # (d, d)
        
        # Off-diagonal squared Frobenius norm
        # Total Frobenius norm squared minus diagonal
        off_diag_loss = (cov ** 2).sum() - (torch.diagonal(cov) ** 2).sum()
        
        # Normalize by number of off-diagonal elements
        covariance_loss = off_diag_loss / (d * (d - 1) + self.eps)
        
        return covariance_loss
    
    def compute_vic_stats(self, support_embeddings, prototypes):
        """
        Compute VIC regularization statistics on support embeddings and prototypes.
        
        Args:
            support_embeddings: (n_way, k_shot, d) support embeddings
            prototypes: (n_way, d) class prototypes
        
        Returns:
            variance_loss: scalar
            covariance_loss: scalar
            stats: dict with additional statistics
        """
        n_way, k_shot, d = support_embeddings.shape
        
        # Concatenate support embeddings with prototypes: [ES, P]
        # Shape: (n_way * (k_shot + 1), d)
        support_flat = support_embeddings.reshape(n_way * k_shot, d)
        all_embeddings = torch.cat([support_flat, prototypes], dim=0)  # (n*(k+1), d)
        
        # Compute variance and covariance losses
        v_loss = self.variance_loss(all_embeddings)
        c_loss = self.covariance_loss(all_embeddings)
        
        # Additional statistics for monitoring
        std = torch.sqrt(all_embeddings.var(dim=0) + self.eps)
        stats = {
            'mean_std': std.mean().item(),
            'min_std': std.min().item(),
            'max_std': std.max().item(),
            'n_samples': all_embeddings.shape[0]
        }
        
        return v_loss, c_loss, stats


class DynamicWeightController(nn.Module):
    """
    Dynamic weight controller for balancing multiple loss terms.
    
    Supports two strategies:
    1. Uncertainty weighting: learn log-variances as parameters
    2. GradNorm: balance gradient norms across tasks
    """
    
    def __init__(self, n_losses=3, method='uncertainty', 
                 init_weights=None, bounds=(0.25, 4.0)):
        """
        Args:
            n_losses: Number of loss terms to balance
            method: 'uncertainty' or 'gradnorm'
            init_weights: Initial weights (e.g., [9.0, 0.5, 0.5] for λI, λV, λC)
            bounds: (min, max) multipliers for clamping weights
        """
        super().__init__()
        self.method = method
        self.n_losses = n_losses
        self.bounds = bounds
        
        if init_weights is None:
            init_weights = [1.0] * n_losses
        
        if method == 'uncertainty':
            # Learn log-variances s_k; weight = exp(-s_k)
            # Initialize so that exp(-s_k) ≈ init_weights[k]
            init_log_vars = [-torch.log(torch.tensor(w)) for w in init_weights]
            self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))
        else:  # gradnorm or fixed
            # Store weights directly as parameters
            self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        
        # Store initial weights for bounding
        self.init_weights = torch.tensor(init_weights, dtype=torch.float32)
    
    def get_weights(self):
        """Get current loss weights."""
        if self.method == 'uncertainty':
            # w_k = exp(-s_k)
            weights = torch.exp(-self.log_vars)
        else:
            weights = self.weights
        
        # Clamp weights to reasonable bounds
        min_weight = self.init_weights * self.bounds[0]
        max_weight = self.init_weights * self.bounds[1]
        weights = torch.clamp(weights, min=min_weight, max=max_weight)
        
        return weights
    
    def compute_total_loss_uncertainty(self, losses):
        """
        Compute total loss with uncertainty weighting.
        Total loss = Σ_k [ L_k * exp(-s_k) + s_k ]
        
        Args:
            losses: list or tensor of loss values [L_I, L_V, L_C]
        
        Returns:
            total_loss: weighted sum of losses
        """
        if isinstance(losses, list):
            losses = torch.stack(losses)
        
        # Uncertainty weighting: L_k * exp(-s_k) + s_k
        weighted_losses = losses * torch.exp(-self.log_vars) + self.log_vars
        total_loss = weighted_losses.sum()
        
        return total_loss
    
    def compute_total_loss_weighted(self, losses):
        """
        Compute simple weighted sum of losses.
        
        Args:
            losses: list or tensor of loss values [L_I, L_V, L_C]
        
        Returns:
            total_loss: weighted sum
        """
        if isinstance(losses, list):
            losses = torch.stack(losses)
        
        weights = self.get_weights()
        total_loss = (losses * weights).sum()
        
        return total_loss
    
    def forward(self, losses):
        """
        Compute total loss with dynamic weighting.
        
        Args:
            losses: list or tensor of loss values [L_I, L_V, L_C]
        
        Returns:
            total_loss: weighted and balanced total loss
        """
        if self.method == 'uncertainty':
            return self.compute_total_loss_uncertainty(losses)
        else:
            return self.compute_total_loss_weighted(losses)
