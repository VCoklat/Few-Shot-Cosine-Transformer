"""
VIC Regularization Module for ProFONet Integration
Implements Variance-Invariance-Covariance regularization for few-shot learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegularization(nn.Module):
    """
    VIC Regularization module implementing:
    - Variance regularization (prevents norm collapse)
    - Invariance regularization (cross-entropy loss)
    - Covariance regularization (prevents representation collapse)
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
        
        Args:
            embeddings: tensor of shape (m, d) where m is number of samples, d is embedding dimension
        Returns:
            variance loss scalar
        """
        # Compute variance along the batch dimension for each feature
        variance = torch.var(embeddings, dim=0, unbiased=False)  # Shape: (d,)
        std = torch.sqrt(variance + self.epsilon)  # Shape: (d,)
        
        # Penalize std that falls below gamma
        v_loss = torch.mean(F.relu(self.gamma - std))
        
        return v_loss
    
    def covariance_loss(self, embeddings):
        """
        Covariance Regularization: prevents representation collapse
        C(E) = (1/(m-1)) * Σ (E_j - Ē)(E_j - Ē)^T
        C_loss = Σ(off_diagonal(C(E))^2) / m
        
        Args:
            embeddings: tensor of shape (m, d)
        Returns:
            covariance loss scalar
        """
        m, d = embeddings.shape
        
        # Center the embeddings
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov_matrix = (embeddings_centered.T @ embeddings_centered) / (m - 1)  # Shape: (d, d)
        
        # Extract off-diagonal elements and compute their squared sum
        # Create mask for off-diagonal elements
        mask = ~torch.eye(d, dtype=torch.bool, device=embeddings.device)
        off_diagonal = cov_matrix[mask]
        
        # Compute loss as mean squared off-diagonal values
        c_loss = torch.mean(off_diagonal ** 2)
        
        return c_loss
    
    def forward(self, embeddings):
        """
        Compute VIC losses (without invariance, which is the classification loss)
        
        Args:
            embeddings: tensor of shape (m, d) containing concatenated support embeddings and prototypes
        Returns:
            dict with 'variance_loss' and 'covariance_loss'
        """
        v_loss = self.variance_loss(embeddings)
        c_loss = self.covariance_loss(embeddings)
        
        return {
            'variance_loss': v_loss,
            'covariance_loss': c_loss
        }


class DynamicVICWeights:
    """
    Dynamic weight adjustment for VIC regularization based on training progress
    """
    
    def __init__(self, lambda_V_base=0.5, lambda_I=9.0, lambda_C_base=0.5):
        self.lambda_V_base = lambda_V_base
        self.lambda_I = lambda_I
        self.lambda_C_base = lambda_C_base
    
    def get_weights(self, current_epoch, total_epochs):
        """
        Compute dynamic weights based on training progress
        
        Args:
            current_epoch: current training epoch (0-indexed)
            total_epochs: total number of training epochs
        Returns:
            dict with 'lambda_V', 'lambda_I', 'lambda_C'
        """
        epoch_ratio = current_epoch / max(total_epochs, 1)
        
        # Increase variance weight during training
        lambda_V = self.lambda_V_base * (1 + 0.3 * epoch_ratio)
        
        # Keep invariance dominant (this is the classification loss weight)
        lambda_I = self.lambda_I
        
        # Decrease covariance weight during training
        lambda_C = self.lambda_C_base * (1 - 0.2 * epoch_ratio)
        
        return {
            'lambda_V': lambda_V,
            'lambda_I': lambda_I,
            'lambda_C': lambda_C
        }
