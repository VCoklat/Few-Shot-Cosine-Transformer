"""
Task-Adaptive Invariance Module

This module implements task-adaptive invariance learning that dynamically
adjusts to different few-shot tasks and multi-scale feature processing
for fine-grained recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAdaptiveInvariance(nn.Module):
    """
    Task-Adaptive Invariance Module
    
    Learns task-specific invariance transformations by applying learnable
    transformations and selecting relevant ones through task-conditioned attention.
    
    Args:
        feature_dim: Dimension of input features
        num_invariance_types: Number of different invariance transformation types
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, feature_dim, num_invariance_types=4, dropout_rate=0.1):
        super(TaskAdaptiveInvariance, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_invariance_types = num_invariance_types
        
        # Learnable invariance transformations
        self.invariance_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, feature_dim)
            ) for _ in range(num_invariance_types)
        ])
        
        # Task-conditioned attention for selecting relevant invariances
        self.task_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_invariance_types),
            nn.Softmax(dim=-1)
        )
        
        # Residual scaling with learnable gamma (initialized small for stability)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, features, task_context=None):
        """
        Apply task-adaptive invariance transformations.
        
        Args:
            features: Input features [batch_size, feature_dim]
            task_context: Optional task-specific context for attention.
                         If None, uses mean of features as context.
        
        Returns:
            Invariance-enhanced features [batch_size, feature_dim]
        """
        batch_size = features.size(0)
        
        # Compute task context if not provided
        if task_context is None:
            task_context = features.mean(dim=0, keepdim=True)  # [1, feature_dim]
        
        # Get attention weights for each invariance type
        attention_weights = self.task_encoder(task_context)  # [1, num_invariance_types]
        
        # Apply all transformations and weight them (vectorized for efficiency)
        # Stack all transformations for batch processing
        transformed_features = torch.stack([
            transform(features) for transform in self.invariance_transforms
        ], dim=0)  # [num_invariance_types, batch_size, feature_dim]
        
        # Weight by attention using einsum for efficient broadcasting
        # attention_weights: [1, num_invariance_types] -> [num_invariance_types, 1, 1]
        weights_expanded = attention_weights.squeeze(0).view(-1, 1, 1)
        weighted_features = transformed_features * weights_expanded
        
        # Aggregate weighted transformations
        aggregated = weighted_features.sum(dim=0)  # [batch_size, feature_dim]
        
        # Residual connection with learnable scaling
        output = features + self.gamma * aggregated
        
        return output


class MultiScaleInvariance(nn.Module):
    """
    Multi-Scale Invariance Module
    
    Processes features at multiple scales to capture both coarse and fine-grained
    patterns, particularly useful for fine-grained recognition tasks like CUB.
    
    Args:
        feature_dim: Dimension of input features
        num_scales: Number of different scales to process
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, feature_dim, num_scales=3, dropout_rate=0.1):
        super(MultiScaleInvariance, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        
        # Multi-scale feature extractors with different receptive fields
        self.scale_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            ) for _ in range(num_scales)
        ])
        
        # Scale-specific attention
        self.scale_attention = nn.Sequential(
            nn.Linear(feature_dim * num_scales, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Residual scaling
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, features):
        """
        Apply multi-scale invariance processing.
        
        Args:
            features: Input features [batch_size, feature_dim]
        
        Returns:
            Multi-scale enhanced features [batch_size, feature_dim]
        """
        # Extract features at different scales
        scale_features = []
        for scale_module in self.scale_modules:
            scale_feat = scale_module(features)
            scale_features.append(scale_feat)
        
        # Concatenate all scale features
        concatenated = torch.cat(scale_features, dim=-1)  # [batch_size, feature_dim * num_scales]
        
        # Compute scale attention
        scale_weights = self.scale_attention(concatenated)  # [batch_size, num_scales]
        
        # Apply attention-weighted aggregation
        weighted_scales = []
        for i, scale_feat in enumerate(scale_features):
            weighted = scale_feat * scale_weights[:, i:i+1]
            weighted_scales.append(weighted)
        
        weighted_concat = torch.cat(weighted_scales, dim=-1)
        
        # Fuse multi-scale features
        fused = self.fusion(weighted_concat)
        
        # Residual connection
        output = features + self.gamma * fused
        
        return output
