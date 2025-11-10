"""
VIC Loss Module for Few-Shot Cosine Transformer

This module implements the Variance-Invariance-Covariance (VIC) loss components
as described in ProFONet paper for enhancing few-shot learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICLoss(nn.Module):
    """
    VIC (Variance-Invariance-Covariance) Loss for few-shot learning.
    
    Combines three loss components:
    1. Invariance Loss (L_I): Standard Cross-Entropy loss for classification
    2. Variance Loss (L_V): Hinge loss on standard deviation to encourage compactness
    3. Covariance Loss (L_C): Covariance regularization to decorrelate features
    
    Args:
        lambda_v (float): Weight for variance loss. Default: 1.0
        lambda_i (float): Weight for invariance (CE) loss. Default: 1.0
        lambda_c (float): Weight for covariance loss. Default: 0.04
        variance_threshold (float): Threshold for variance hinge loss. Default: 1.0
        epsilon (float): Small constant for numerical stability. Default: 1e-4
    """
    
    def __init__(self, lambda_v=1.0, lambda_i=1.0, lambda_c=0.04, 
                 variance_threshold=1.0, epsilon=1e-4):
        super(VICLoss, self).__init__()
        self.lambda_v = lambda_v
        self.lambda_i = lambda_i
        self.lambda_c = lambda_c
        self.variance_threshold = variance_threshold
        self.epsilon = epsilon
        self.ce_loss = nn.CrossEntropyLoss()
    
    def invariance_loss(self, predictions, targets):
        """
        Compute invariance loss (standard cross-entropy).
        
        Args:
            predictions: Model predictions (logits), shape (batch, num_classes)
            targets: Ground truth labels, shape (batch,)
            
        Returns:
            Cross-entropy loss
        """
        return self.ce_loss(predictions, targets)
    
    def variance_loss(self, embeddings, n_way, k_shot):
        """
        Compute variance loss using hinge loss on standard deviation.
        Encourages compact representations for each class.
        
        Args:
            embeddings: Support set embeddings, shape (n_way * k_shot, feature_dim)
            n_way: Number of classes
            k_shot: Number of samples per class
            
        Returns:
            Variance loss
        """
        # Reshape embeddings to (n_way, k_shot, feature_dim)
        embeddings = embeddings.view(n_way, k_shot, -1)
        
        # Compute standard deviation for each class
        # std shape: (n_way, feature_dim)
        std = torch.std(embeddings, dim=1, unbiased=False)
        
        # Apply hinge loss: max(0, threshold - std)
        # We want std to be at least threshold (compact clusters)
        hinge = F.relu(self.variance_threshold - std)
        
        # Average over all classes and dimensions
        loss = torch.mean(hinge)
        
        return loss
    
    def covariance_loss(self, embeddings):
        """
        Compute covariance loss to decorrelate feature dimensions.
        Prevents informational collapse by encouraging diverse feature representations.
        
        Args:
            embeddings: Support set embeddings, shape (batch, feature_dim)
            
        Returns:
            Covariance loss
        """
        batch_size, feature_dim = embeddings.shape
        
        # Center the embeddings (zero mean)
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix: (feature_dim, feature_dim)
        # Cov = (X^T X) / (N - 1)
        cov_matrix = torch.matmul(
            embeddings_centered.T, embeddings_centered
        ) / (batch_size - 1 + self.epsilon)
        
        # Off-diagonal elements should be zero (decorrelation)
        # Create mask for off-diagonal elements
        off_diagonal_mask = ~torch.eye(feature_dim, dtype=torch.bool, device=embeddings.device)
        
        # Sum of squared off-diagonal elements
        off_diagonal_cov = cov_matrix[off_diagonal_mask]
        loss = torch.sum(off_diagonal_cov ** 2) / (feature_dim * (feature_dim - 1))
        
        return loss
    
    def forward(self, predictions, targets, support_embeddings, n_way, k_shot):
        """
        Compute total VIC loss.
        
        Args:
            predictions: Model predictions (logits), shape (n_query, n_way)
            targets: Ground truth labels for query samples, shape (n_query,)
            support_embeddings: Support set embeddings, shape (n_way * k_shot, feature_dim)
            n_way: Number of classes
            k_shot: Number of samples per class
            
        Returns:
            Dictionary containing:
                - 'total': Total weighted loss
                - 'invariance': Invariance loss value
                - 'variance': Variance loss value
                - 'covariance': Covariance loss value
        """
        # Compute individual losses
        l_i = self.invariance_loss(predictions, targets)
        l_v = self.variance_loss(support_embeddings, n_way, k_shot)
        l_c = self.covariance_loss(support_embeddings)
        
        # Combine losses with weights
        total_loss = (self.lambda_i * l_i + 
                     self.lambda_v * l_v + 
                     self.lambda_c * l_c)
        
        return {
            'total': total_loss,
            'invariance': l_i.item(),
            'variance': l_v.item(),
            'covariance': l_c.item()
        }
