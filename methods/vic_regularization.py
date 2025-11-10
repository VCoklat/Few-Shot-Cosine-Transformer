"""
VIC (Variance-Invariance-Covariance) Regularization Module
Based on VICReg and ProFONet papers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VICRegularization(nn.Module):
    """
    Implements VIC regularization for few-shot learning:
    - Variance: Encourages feature dimensions to have std close to 1
    - Invariance: Classification loss (handled externally via Mahalanobis)
    - Covariance: Decorrelates features across dimensions
    """
    def __init__(self, feature_dim, epsilon=1e-4, target_std=1.0):
        super(VICRegularization, self).__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.target_std = target_std
        
    def variance_loss(self, embeddings):
        """
        Compute variance loss: hinge on per-dimension std toward target σ=1
        V = (1/d) Σ_j max(0, σ_j - 1 - ε) where σ_j = sqrt(Var(E[:, j]) + ε)
        
        Args:
            embeddings: (N, d) tensor of embeddings
        Returns:
            variance_loss: scalar
        """
        # Compute std per dimension
        std = torch.sqrt(embeddings.var(dim=0) + self.epsilon)
        
        # Hinge loss: penalize std below (target - epsilon)
        variance_loss = torch.mean(F.relu(self.target_std - std - self.epsilon))
        
        return variance_loss
    
    def covariance_loss(self, embeddings):
        """
        Compute covariance loss: off-diagonal squared Frobenius norm
        C = (1/d) Σ_{i≠j} C_ij²
        
        Args:
            embeddings: (N, d) tensor of embeddings
        Returns:
            covariance_loss: scalar
        """
        N, d = embeddings.shape
        
        # Normalize embeddings (zero mean, unit variance per dimension)
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        embeddings = embeddings / (embeddings.std(dim=0, keepdim=True) + self.epsilon)
        
        # Compute covariance matrix
        cov_matrix = (embeddings.T @ embeddings) / (N - 1)
        
        # Off-diagonal elements
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        
        # Squared Frobenius norm of off-diagonal
        covariance_loss = (off_diag ** 2).sum() / d
        
        return covariance_loss
    
    def forward(self, support_embeddings, prototypes):
        """
        Compute VIC regularization terms on support embeddings and prototypes
        
        Args:
            support_embeddings: (n_way, k_shot, d) or (n_way * k_shot, d)
            prototypes: (n_way, d)
        Returns:
            variance_loss, covariance_loss
        """
        # Flatten support embeddings if needed
        if support_embeddings.dim() == 3:
            n_way, k_shot, d = support_embeddings.shape
            support_embeddings = support_embeddings.reshape(n_way * k_shot, d)
        
        # Concatenate support embeddings and prototypes
        all_embeddings = torch.cat([support_embeddings, prototypes], dim=0)
        
        # Compute losses
        variance_loss = self.variance_loss(all_embeddings)
        covariance_loss = self.covariance_loss(all_embeddings)
        
        return variance_loss, covariance_loss


class MahalanobisClassifier(nn.Module):
    """
    Mahalanobis distance-based classifier with shrinkage covariance
    """
    def __init__(self, shrinkage_param=0.1):
        super(MahalanobisClassifier, self).__init__()
        self.shrinkage_param = shrinkage_param
        
    def compute_shrinkage_covariance(self, embeddings, shrinkage=None):
        """
        Compute shrinkage covariance: Σ = (1-α)S + αI
        
        Args:
            embeddings: (k, d) tensor
            shrinkage: optional shrinkage parameter, defaults to d/(k+d)
        Returns:
            cov_matrix: (d, d) shrinkage covariance matrix
        """
        k, d = embeddings.shape
        
        if shrinkage is None:
            # Ledoit-Wolf optimal shrinkage for small samples
            shrinkage = d / (k + d)
        
        # Compute sample covariance
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        sample_cov = (embeddings_centered.T @ embeddings_centered) / (k - 1)
        
        # Apply shrinkage toward identity
        shrinkage_cov = (1 - shrinkage) * sample_cov + shrinkage * torch.eye(d, device=embeddings.device)
        
        return shrinkage_cov
    
    def mahalanobis_distance(self, query, prototype, cov_inv):
        """
        Compute Mahalanobis distance: D(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
        
        Args:
            query: (d,) or (n_query, d)
            prototype: (d,)
            cov_inv: (d, d) inverse covariance
        Returns:
            distances: scalar or (n_query,)
        """
        diff = query - prototype
        
        if diff.dim() == 1:
            # Single query
            dist = diff @ cov_inv @ diff
        else:
            # Multiple queries: (n_query, d) @ (d, d) @ (d, n_query)
            dist = torch.sum((diff @ cov_inv) * diff, dim=1)
        
        return dist
    
    def forward(self, queries, support_embeddings, prototypes, shrinkage=None):
        """
        Compute Mahalanobis distances from queries to each class prototype
        
        Args:
            queries: (n_query, d)
            support_embeddings: (n_way, k_shot, d)
            prototypes: (n_way, d)
            shrinkage: optional shrinkage parameter
        Returns:
            distances: (n_query, n_way) - negative distances for logits
        """
        n_way, k_shot, d = support_embeddings.shape
        n_query = queries.shape[0]
        
        if shrinkage is None:
            shrinkage = self.shrinkage_param
        
        # Compute distances for each class
        distances = []
        for c in range(n_way):
            # Get support embeddings for this class
            class_support = support_embeddings[c]  # (k_shot, d)
            
            # Compute shrinkage covariance
            cov_matrix = self.compute_shrinkage_covariance(class_support, shrinkage)
            
            # Compute inverse using Cholesky decomposition for stability
            try:
                L = torch.linalg.cholesky(cov_matrix)
                cov_inv = torch.cholesky_inverse(L)
            except RuntimeError:
                # Fallback to pseudo-inverse if Cholesky fails
                cov_inv = torch.linalg.pinv(cov_matrix)
            
            # Compute Mahalanobis distances
            class_dist = self.mahalanobis_distance(queries, prototypes[c], cov_inv)
            distances.append(class_dist)
        
        # Stack distances: (n_way, n_query) -> (n_query, n_way)
        distances = torch.stack(distances, dim=1)
        
        # Return negative distances as logits (higher = closer)
        return -distances


class DynamicWeightController(nn.Module):
    """
    Dynamic weight controller for balancing VIC loss terms
    Supports three strategies:
    1. Uncertainty weighting (learnable log-variances)
    2. GradNorm
    3. Stats-driven
    """
    def __init__(self, strategy='uncertainty', initial_weights=None):
        super(DynamicWeightController, self).__init__()
        self.strategy = strategy
        
        if initial_weights is None:
            initial_weights = {'invariance': 9.0, 'variance': 0.5, 'covariance': 0.5}
        
        if strategy == 'uncertainty':
            # Learnable log-variances for uncertainty weighting
            # Loss = Σ_k [ L_k * exp(-s_k) + s_k ]
            self.log_var_I = nn.Parameter(torch.tensor(0.0))
            self.log_var_V = nn.Parameter(torch.tensor(0.0))
            self.log_var_C = nn.Parameter(torch.tensor(0.0))
        elif strategy == 'gradnorm':
            # Fixed weights updated via gradient norm balancing
            self.register_buffer('lambda_I', torch.tensor(initial_weights['invariance']))
            self.register_buffer('lambda_V', torch.tensor(initial_weights['variance']))
            self.register_buffer('lambda_C', torch.tensor(initial_weights['covariance']))
            self.alpha = 1.0  # GradNorm alpha parameter
        else:  # stats-driven
            # Weights computed from loss statistics
            self.register_buffer('lambda_I', torch.tensor(initial_weights['invariance']))
            self.register_buffer('lambda_V', torch.tensor(initial_weights['variance']))
            self.register_buffer('lambda_C', torch.tensor(initial_weights['covariance']))
    
    def get_weights(self):
        """Get current loss weights"""
        if self.strategy == 'uncertainty':
            # Uncertainty weighting: weight = exp(-log_var)
            return {
                'invariance': torch.exp(-self.log_var_I),
                'variance': torch.exp(-self.log_var_V),
                'covariance': torch.exp(-self.log_var_C)
            }
        else:
            return {
                'invariance': self.lambda_I,
                'variance': self.lambda_V,
                'covariance': self.lambda_C
            }
    
    def compute_weighted_loss(self, loss_I, loss_V, loss_C):
        """
        Compute weighted loss with regularization
        
        Args:
            loss_I: invariance (classification) loss
            loss_V: variance loss
            loss_C: covariance loss
        Returns:
            total_loss: weighted sum
            individual_losses: dict
        """
        if self.strategy == 'uncertainty':
            # Uncertainty weighting: L_k * exp(-s_k) + s_k
            total_loss = (
                loss_I * torch.exp(-self.log_var_I) + self.log_var_I +
                loss_V * torch.exp(-self.log_var_V) + self.log_var_V +
                loss_C * torch.exp(-self.log_var_C) + self.log_var_C
            )
        else:
            weights = self.get_weights()
            total_loss = (
                weights['invariance'] * loss_I +
                weights['variance'] * loss_V +
                weights['covariance'] * loss_C
            )
        
        return total_loss, {
            'loss_I': loss_I.item(),
            'loss_V': loss_V.item(),
            'loss_C': loss_C.item(),
            'weight_I': self.get_weights()['invariance'].item(),
            'weight_V': self.get_weights()['variance'].item(),
            'weight_C': self.get_weights()['covariance'].item()
        }
    
    def update_weights_gradnorm(self, losses, shared_params, alpha=1.0):
        """
        Update weights using GradNorm algorithm
        
        Args:
            losses: dict with 'invariance', 'variance', 'covariance'
            shared_params: shared parameters to compute gradients w.r.t.
            alpha: GradNorm alpha parameter (0.5-1.5)
        """
        if self.strategy != 'gradnorm':
            return
        
        # Compute gradient norms for each loss
        # This would be called after backward pass
        # Implementation skipped for brevity as it requires careful integration
        pass
    
    def update_weights_stats(self, loss_I, loss_V, loss_C, lambda_sum=10.0):
        """
        Update weights based on loss statistics
        
        Args:
            loss_I: current invariance loss
            loss_V: current variance loss
            loss_C: current covariance loss
            lambda_sum: target sum of weights
        """
        if self.strategy != 'stats':
            return
        
        # Normalize weights proportional to loss magnitudes
        total = loss_I + loss_V + loss_C + 1e-8
        self.lambda_I = lambda_sum * (loss_I / total)
        self.lambda_V = lambda_sum * (loss_V / total)
        self.lambda_C = lambda_sum * (loss_C / total)
        
        # Clamp to reasonable bounds
        self.lambda_I = torch.clamp(self.lambda_I, 0.25 * 9.0, 4.0 * 9.0)
        self.lambda_V = torch.clamp(self.lambda_V, 0.25 * 0.5, 4.0 * 0.5)
        self.lambda_C = torch.clamp(self.lambda_C, 0.25 * 0.5, 4.0 * 0.5)
