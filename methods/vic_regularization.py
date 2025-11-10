"""
ProFONet VIC (Variance-Invariance-Covariance) Regularization
with Dynamic Weight Adjustment for Few-Shot Learning

This module implements the VIC regularization from ProFONet paper,
designed to create robust and discriminative feature spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegularization(nn.Module):
    """
    VIC Regularization module that computes variance, invariance, and covariance losses
    for feature embeddings to encourage spread, consistency, and decorrelation.
    
    Args:
        lambda_v: Initial weight for variance loss
        lambda_i: Initial weight for invariance loss
        lambda_c: Initial weight for covariance loss
        epsilon: Minimum variance threshold
        dynamic_weights: Whether to use dynamic weight adjustment
        alpha: Learning rate for dynamic weight updates
    """
    
    def __init__(self, lambda_v=1.0, lambda_i=1.0, lambda_c=0.04, 
                 epsilon=1e-4, dynamic_weights=True, alpha=0.001):
        super(VICRegularization, self).__init__()
        
        # Initialize weights as learnable parameters if dynamic, else as fixed
        if dynamic_weights:
            self.lambda_v = nn.Parameter(torch.tensor(lambda_v))
            self.lambda_i = nn.Parameter(torch.tensor(lambda_i))
            self.lambda_c = nn.Parameter(torch.tensor(lambda_c))
        else:
            self.register_buffer('lambda_v', torch.tensor(lambda_v))
            self.register_buffer('lambda_i', torch.tensor(lambda_i))
            self.register_buffer('lambda_c', torch.tensor(lambda_c))
        
        self.epsilon = epsilon
        self.dynamic_weights = dynamic_weights
        self.alpha = alpha
        
        # Running statistics for dynamic weight adjustment
        self.register_buffer('running_v_loss', torch.tensor(0.0))
        self.register_buffer('running_i_loss', torch.tensor(0.0))
        self.register_buffer('running_c_loss', torch.tensor(0.0))
        self.register_buffer('update_count', torch.tensor(0))
        
    def variance_loss(self, embeddings):
        """
        Variance loss: Encourages spread within each feature dimension
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim) or (n_way, k_shot, embedding_dim)
        
        Returns:
            Variance loss scalar
        """
        # Flatten if needed: (n_way, k_shot, d) -> (n_way * k_shot, d)
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        # Compute variance per dimension
        var = torch.var(embeddings, dim=0, unbiased=True)
        
        # Penalize dimensions with variance below epsilon
        v_loss = torch.mean(F.relu(self.epsilon - var))
        
        return v_loss
    
    def invariance_loss(self, embeddings, labels=None):
        """
        Invariance loss: Encourages embeddings within the same class to be similar
        
        Args:
            embeddings: Tensor of shape (n_way, k_shot, embedding_dim)
            labels: Optional labels for computing class-wise invariance
        
        Returns:
            Invariance loss scalar
        """
        if embeddings.dim() == 2:
            # If already flattened, can't compute class-wise invariance
            # Return zero loss
            return torch.tensor(0.0, device=embeddings.device)
        
        # embeddings shape: (n_way, k_shot, d)
        n_way, k_shot, d = embeddings.shape
        
        # Compute class-wise mean
        class_means = embeddings.mean(dim=1, keepdim=True)  # (n_way, 1, d)
        
        # Mean squared distance from class mean
        i_loss = torch.mean((embeddings - class_means) ** 2)
        
        return i_loss
    
    def covariance_loss(self, embeddings):
        """
        Covariance loss: Encourages decorrelation between feature dimensions
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim) or (n_way, k_shot, embedding_dim)
        
        Returns:
            Covariance loss scalar
        """
        # Flatten if needed: (n_way, k_shot, d) -> (n_way * k_shot, d)
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        batch_size, d = embeddings.shape
        
        # Center the embeddings
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = (embeddings_centered.T @ embeddings_centered) / (batch_size - 1)
        
        # Sum of squared off-diagonal elements
        # We want off-diagonal to be close to zero (decorrelated features)
        off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        c_loss = torch.sum(off_diag ** 2) / d
        
        return c_loss
    
    def forward(self, embeddings, labels=None):
        """
        Compute total VIC loss
        
        Args:
            embeddings: Support set embeddings, shape (n_way, k_shot, embedding_dim) or (batch, d)
            labels: Optional labels for class-wise losses
        
        Returns:
            total_loss: Weighted combination of V, I, C losses
            loss_dict: Dictionary with individual loss components
        """
        # Compute individual losses
        v_loss = self.variance_loss(embeddings)
        i_loss = self.invariance_loss(embeddings, labels)
        c_loss = self.covariance_loss(embeddings)
        
        # Update running statistics for dynamic weights
        if self.training and self.dynamic_weights:
            momentum = 0.9
            self.running_v_loss = momentum * self.running_v_loss + (1 - momentum) * v_loss.detach()
            self.running_i_loss = momentum * self.running_i_loss + (1 - momentum) * i_loss.detach()
            self.running_c_loss = momentum * self.running_c_loss + (1 - momentum) * c_loss.detach()
            self.update_count += 1
        
        # Compute weighted loss
        total_loss = (self.lambda_v * v_loss + 
                     self.lambda_i * i_loss + 
                     self.lambda_c * c_loss)
        
        # Return total loss and individual components
        loss_dict = {
            'vic_total': total_loss,
            'vic_variance': v_loss,
            'vic_invariance': i_loss,
            'vic_covariance': c_loss,
            'lambda_v': self.lambda_v.item() if isinstance(self.lambda_v, torch.Tensor) else self.lambda_v,
            'lambda_i': self.lambda_i.item() if isinstance(self.lambda_i, torch.Tensor) else self.lambda_i,
            'lambda_c': self.lambda_c.item() if isinstance(self.lambda_c, torch.Tensor) else self.lambda_c,
        }
        
        return total_loss, loss_dict
    
    def update_dynamic_weights(self):
        """
        Update dynamic weights based on running loss statistics
        This should be called after optimizer.step()
        """
        if not self.dynamic_weights or self.update_count < 10:
            return
        
        # Normalize running losses to prevent extreme weight adjustments
        total_loss = self.running_v_loss + self.running_i_loss + self.running_c_loss
        if total_loss > 0:
            # Adjust weights inversely proportional to loss magnitude
            # Losses that are too high get reduced weight, losses that are too low get increased weight
            target_ratio = 1.0 / 3.0  # Equal contribution target
            
            v_ratio = self.running_v_loss / (total_loss + 1e-8)
            i_ratio = self.running_i_loss / (total_loss + 1e-8)
            c_ratio = self.running_c_loss / (total_loss + 1e-8)
            
            # Adjust weights with small learning rate
            with torch.no_grad():
                if self.dynamic_weights:
                    self.lambda_v.data = torch.clamp(
                        self.lambda_v + self.alpha * (target_ratio - v_ratio),
                        min=0.01, max=10.0
                    )
                    self.lambda_i.data = torch.clamp(
                        self.lambda_i + self.alpha * (target_ratio - i_ratio),
                        min=0.01, max=10.0
                    )
                    self.lambda_c.data = torch.clamp(
                        self.lambda_c + self.alpha * (target_ratio - c_ratio),
                        min=0.001, max=1.0  # Covariance typically needs smaller weights
                    )
    
    def get_weights(self):
        """Return current weight values"""
        return {
            'lambda_v': self.lambda_v.item() if isinstance(self.lambda_v, torch.Tensor) else self.lambda_v,
            'lambda_i': self.lambda_i.item() if isinstance(self.lambda_i, torch.Tensor) else self.lambda_i,
            'lambda_c': self.lambda_c.item() if isinstance(self.lambda_c, torch.Tensor) else self.lambda_c,
        }
