"""
Mahalanobis distance classifier with shrinkage covariance estimation.
Based on ProFONet and Mahalanobis-FSL papers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MahalanobisClassifier(nn.Module):
    """
    Classifier using Mahalanobis distance with shrinkage covariance.
    
    For each class c with support embeddings ES[c]:
    - Compute sample covariance Sc
    - Apply shrinkage: Σc = (1-α)·Sc + α·I
    - Distance: D[x,c] = (x - P[c])ᵀ Σc⁻¹ (x - P[c])
    - Logits: -D[x,c]
    """
    
    def __init__(self, shrinkage_alpha=None, eps=1e-4):
        """
        Args:
            shrinkage_alpha: Fixed alpha or None for adaptive (d/(k+d))
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.shrinkage_alpha = shrinkage_alpha
        self.eps = eps
    
    def compute_shrinkage_covariance(self, support_embeddings, alpha=None):
        """
        Compute shrinkage covariance for support embeddings.
        
        Args:
            support_embeddings: (k, d) tensor of k support samples with d dimensions
            alpha: Shrinkage parameter, if None use adaptive α = d/(k+d)
        
        Returns:
            inv_cov: (d, d) inverse covariance matrix
        """
        k, d = support_embeddings.shape
        
        # Compute sample covariance
        mean = support_embeddings.mean(dim=0, keepdim=True)
        centered = support_embeddings - mean
        
        # Use more numerically stable covariance computation
        # cov = (centered.T @ centered) / (k - 1 + self.eps)
        cov = torch.mm(centered.T, centered) / (k - 1 + self.eps)
        
        # Determine shrinkage alpha
        if alpha is None:
            if self.shrinkage_alpha is None:
                alpha = d / (k + d)
            else:
                alpha = self.shrinkage_alpha
        
        # Apply shrinkage: Σ = (1-α)·S + α·I
        # Use in-place operations to save memory
        shrunk_cov = cov.mul(1 - alpha)
        shrunk_cov.diagonal().add_(alpha)
        
        # Compute inverse using Cholesky decomposition for stability
        try:
            L = torch.linalg.cholesky(shrunk_cov)
            inv_cov = torch.cholesky_inverse(L)
        except RuntimeError as e:
            # Fallback to adding more regularization
            shrunk_cov.diagonal().add_(self.eps)
            try:
                inv_cov = torch.linalg.inv(shrunk_cov)
            except RuntimeError:
                # If still failing, use pseudo-inverse as last resort
                inv_cov = torch.linalg.pinv(shrunk_cov)
        
        return inv_cov
    
    def mahalanobis_distance(self, query, prototype, inv_cov):
        """
        Compute Mahalanobis distance: D = (x - μ)ᵀ Σ⁻¹ (x - μ)
        
        Args:
            query: (nq, d) query embeddings
            prototype: (d,) class prototype
            inv_cov: (d, d) inverse covariance matrix
        
        Returns:
            distances: (nq,) Mahalanobis distances
        """
        # Center query around prototype
        diff = query - prototype.unsqueeze(0)  # (nq, d)
        
        # Compute Mahalanobis distance: (x-μ)ᵀ Σ⁻¹ (x-μ)
        # = sum((x-μ) * (Σ⁻¹ @ (x-μ).T).T)
        mahal_component = diff @ inv_cov  # (nq, d)
        distances = (mahal_component * diff).sum(dim=-1)  # (nq,)
        
        return distances
    
    def forward(self, query_embeddings, support_embeddings, prototypes):
        """
        Compute classification logits using Mahalanobis distance.
        
        Args:
            query_embeddings: (nq, d) query embeddings after attention
            support_embeddings: (n_way, k_shot, d) support embeddings
            prototypes: (n_way, d) class prototypes
        
        Returns:
            logits: (nq, n_way) classification logits as -D(x, c)
        """
        n_way, k_shot, d = support_embeddings.shape
        nq = query_embeddings.shape[0]
        
        # Compute inverse covariances for each class
        inv_covs = []
        for c in range(n_way):
            inv_cov = self.compute_shrinkage_covariance(support_embeddings[c])
            inv_covs.append(inv_cov)
        
        # Compute distances for all queries to all classes
        logits = torch.zeros(nq, n_way, device=query_embeddings.device)
        for c in range(n_way):
            distances = self.mahalanobis_distance(
                query_embeddings, prototypes[c], inv_covs[c]
            )
            # Negative distance as logits (higher similarity = lower distance)
            logits[:, c] = -distances
        
        return logits
