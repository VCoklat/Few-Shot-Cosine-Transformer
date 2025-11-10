"""
VIC (Variance-Invariance-Covariance) Regularization Module
Based on ProFONet framework with dynamic weight adjustment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VICRegularization(nn.Module):
    """
    VIC Regularization with dynamic weighting for few-shot learning.
    
    Implements:
    - Variance loss: Encourages spread within class
    - Invariance loss: Mean squared distance between embeddings within class
    - Covariance loss: Off-diagonal covariance, encourages orthogonal features
    - Dynamic weight adjustment: Adapts loss weights during training
    """
    
    def __init__(self, 
                 lambda_v=1.0, 
                 lambda_i=1.0, 
                 lambda_c=1.0,
                 epsilon=1e-4,
                 alpha=0.001,
                 min_weight=0.01,
                 max_weight=10.0):
        """
        Args:
            lambda_v: Initial weight for variance loss
            lambda_i: Initial weight for invariance loss
            lambda_c: Initial weight for covariance loss
            epsilon: Minimum variance threshold
            alpha: Learning rate for dynamic weight updates
            min_weight: Minimum allowed weight value
            max_weight: Maximum allowed weight value
        """
        super(VICRegularization, self).__init__()
        
        # Initialize dynamic weights as learnable parameters
        self.lambda_v = nn.Parameter(torch.tensor(lambda_v))
        self.lambda_i = nn.Parameter(torch.tensor(lambda_i))
        self.lambda_c = nn.Parameter(torch.tensor(lambda_c))
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Running statistics for dynamic weight adjustment
        self.register_buffer('running_v_loss', torch.tensor(0.0))
        self.register_buffer('running_i_loss', torch.tensor(0.0))
        self.register_buffer('running_c_loss', torch.tensor(0.0))
        self.register_buffer('update_count', torch.tensor(0))
        
    def variance_loss(self, embeddings):
        """
        Variance loss: Encourages embeddings to have sufficient variance.
        
        Args:
            embeddings: Tensor of shape (batch, n_way * k_shot, dim) or (batch, dim)
            
        Returns:
            Scalar loss value
        """
        # Reshape if needed to (batch, dim)
        if embeddings.dim() == 3:
            embeddings = embeddings.view(-1, embeddings.size(-1))
        
        # Compute variance along batch dimension for each feature
        var = torch.var(embeddings, dim=0, unbiased=False)
        
        # Encourage variance to be above epsilon
        loss = torch.mean(F.relu(self.epsilon - var))
        
        return loss
    
    def invariance_loss(self, embeddings, labels=None):
        """
        Invariance loss: Mean squared distance between embeddings within the same class.
        
        Args:
            embeddings: Tensor of shape (n_way, k_shot, dim)
            labels: Optional class labels (not used if embeddings are pre-grouped)
            
        Returns:
            Scalar loss value
        """
        if embeddings.dim() != 3:
            # If not properly shaped, return zero loss
            return torch.tensor(0.0, device=embeddings.device)
        
        n_way, k_shot, dim = embeddings.shape
        
        # Compute mean embedding for each class
        class_means = embeddings.mean(dim=1, keepdim=True)  # (n_way, 1, dim)
        
        # Compute MSE between each sample and its class mean
        mse = torch.mean((embeddings - class_means) ** 2)
        
        return mse
    
    def covariance_loss(self, embeddings):
        """
        Covariance loss: Off-diagonal sum of covariance matrix.
        Encourages decorrelated (orthogonal) features.
        
        Args:
            embeddings: Tensor of shape (batch, dim)
            
        Returns:
            Scalar loss value
        """
        # Reshape if needed to (batch, dim)
        if embeddings.dim() == 3:
            embeddings = embeddings.view(-1, embeddings.size(-1))
        
        batch_size, dim = embeddings.shape
        
        # Center the embeddings
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = torch.mm(embeddings_centered.T, embeddings_centered) / (batch_size - 1)
        
        # Off-diagonal covariance
        off_diag = cov - torch.diag(torch.diag(cov))
        
        # Sum of squared off-diagonal elements
        loss = torch.sum(off_diag ** 2)
        
        return loss
    
    def forward(self, support_embeddings, query_embeddings=None):
        """
        Compute total VIC loss with dynamic weighting.
        
        Args:
            support_embeddings: Support set embeddings (n_way, k_shot, dim)
            query_embeddings: Optional query embeddings for transductive learning
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        # Compute individual losses on support set
        v_loss = self.variance_loss(support_embeddings)
        i_loss = self.invariance_loss(support_embeddings)
        c_loss = self.covariance_loss(support_embeddings)
        
        # If query embeddings provided, add their contribution (transductive)
        if query_embeddings is not None:
            v_loss += self.variance_loss(query_embeddings)
            # Covariance on queries
            if query_embeddings.dim() == 3:
                c_loss += self.covariance_loss(query_embeddings.view(-1, query_embeddings.size(-1)))
            else:
                c_loss += self.covariance_loss(query_embeddings)
            
            # Average the losses
            v_loss /= 2.0
            c_loss /= 2.0
        
        # Clamp weights to prevent extreme values
        lambda_v = torch.clamp(self.lambda_v, self.min_weight, self.max_weight)
        lambda_i = torch.clamp(self.lambda_i, self.min_weight, self.max_weight)
        lambda_c = torch.clamp(self.lambda_c, self.min_weight, self.max_weight)
        
        # Compute weighted total loss
        total_loss = lambda_v * v_loss + lambda_i * i_loss + lambda_c * c_loss
        
        # Update running statistics (for monitoring)
        if self.training:
            self.running_v_loss = 0.9 * self.running_v_loss + 0.1 * v_loss.detach()
            self.running_i_loss = 0.9 * self.running_i_loss + 0.1 * i_loss.detach()
            self.running_c_loss = 0.9 * self.running_c_loss + 0.1 * c_loss.detach()
            self.update_count += 1
        
        return {
            'total': total_loss,
            'variance': v_loss,
            'invariance': i_loss,
            'covariance': c_loss,
            'lambda_v': lambda_v.detach(),
            'lambda_i': lambda_i.detach(),
            'lambda_c': lambda_c.detach()
        }
    
    def update_dynamic_weights(self, v_loss, i_loss, c_loss):
        """
        Update dynamic weights based on gradient information.
        This is called after backward pass.
        
        Args:
            v_loss: Variance loss value
            i_loss: Invariance loss value
            c_loss: Covariance loss value
        """
        if not self.training:
            return
        
        # Adaptive weight adjustment based on loss magnitudes
        # If a loss is too high, increase its weight
        # If a loss is too low, decrease its weight
        total = v_loss + i_loss + c_loss + 1e-8
        
        # Target: balance the losses
        with torch.no_grad():
            # Normalize by total to get relative magnitudes
            v_ratio = v_loss / total
            i_ratio = i_loss / total
            c_ratio = c_loss / total
            
            # Adjust weights inversely proportional to their contribution
            # (losses that are too small get higher weight, losses too large get lower weight)
            target = 1.0 / 3.0  # Target equal contribution
            
            self.lambda_v.data += self.alpha * (target - v_ratio) * self.lambda_v.data
            self.lambda_i.data += self.alpha * (target - i_ratio) * self.lambda_i.data
            self.lambda_c.data += self.alpha * (target - c_ratio) * self.lambda_c.data
            
            # Clamp weights
            self.lambda_v.data = torch.clamp(self.lambda_v.data, self.min_weight, self.max_weight)
            self.lambda_i.data = torch.clamp(self.lambda_i.data, self.min_weight, self.max_weight)
            self.lambda_c.data = torch.clamp(self.lambda_c.data, self.min_weight, self.max_weight)
    
    def get_weight_stats(self):
        """Return current weight statistics for monitoring."""
        return {
            'lambda_v': self.lambda_v.item(),
            'lambda_i': self.lambda_i.item(),
            'lambda_c': self.lambda_c.item(),
            'running_v_loss': self.running_v_loss.item(),
            'running_i_loss': self.running_i_loss.item(),
            'running_c_loss': self.running_c_loss.item(),
            'update_count': self.update_count.item()
        }
