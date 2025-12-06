"""
Medical Image Invariance Module

This module implements domain-specific invariance for medical imaging tasks,
particularly dermoscopy images (HAM10000), with pathways for color/intensity,
texture, and shape invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalImageInvariance(nn.Module):
    """
    Medical Image Invariance Module
    
    Specialized module for medical imaging that processes features through
    multiple pathways to achieve invariance to color/intensity variations,
    while preserving critical texture and shape information.
    
    Args:
        feature_dim: Dimension of input features
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, feature_dim, dropout_rate=0.1):
        super(MedicalImageInvariance, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Color/Intensity invariance pathway
        # Handles variations in lighting, staining, and color balance
        self.color_pathway = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Texture-focused pathway (critical for dermoscopy)
        # Preserves fine-grained texture patterns important for diagnosis
        self.texture_pathway = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Shape-focused pathway
        # Preserves structural and morphological features
        self.shape_pathway = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Adaptive fusion mechanism
        # Learns to weight different pathways based on input
        self.fusion_attention = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Residual scaling
        self.gamma = nn.Parameter(torch.ones(1) * 0.15)
        
    def forward(self, features):
        """
        Apply medical image invariance processing.
        
        Args:
            features: Input features [batch_size, feature_dim]
        
        Returns:
            Invariance-enhanced features [batch_size, feature_dim]
        """
        # Process through each pathway
        color_features = self.color_pathway(features)  # [batch_size, feature_dim]
        texture_features = self.texture_pathway(features)  # [batch_size, feature_dim]
        shape_features = self.shape_pathway(features)  # [batch_size, feature_dim]
        
        # Concatenate pathway outputs
        concatenated = torch.cat([color_features, texture_features, shape_features], dim=-1)
        
        # Compute adaptive fusion weights
        fusion_weights = self.fusion_attention(concatenated)  # [batch_size, 3]
        
        # Apply weighted fusion
        weighted_color = color_features * fusion_weights[:, 0:1]
        weighted_texture = texture_features * fusion_weights[:, 1:2]
        weighted_shape = shape_features * fusion_weights[:, 2:3]
        
        weighted_concat = torch.cat([weighted_color, weighted_texture, weighted_shape], dim=-1)
        
        # Final projection
        fused = self.output_projection(weighted_concat)
        
        # Residual connection
        output = features + self.gamma * fused
        
        # Normalize
        output = F.normalize(output, p=2, dim=1)
        
        return output


class ContrastiveInvarianceLoss(nn.Module):
    """
    Contrastive Invariance Loss
    
    Additional loss to enforce invariance learning by pulling together
    different augmented views while pushing apart different classes.
    
    Args:
        temperature: Temperature parameter for contrastive loss
    """
    
    def __init__(self, temperature=0.07):
        super(ContrastiveInvarianceLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features1, features2, labels=None):
        """
        Compute contrastive invariance loss.
        
        Args:
            features1: First set of features [batch_size, feature_dim]
            features2: Second set of features (augmented/transformed) [batch_size, feature_dim]
            labels: Optional class labels [batch_size]
        
        Returns:
            Contrastive loss value
        """
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        batch_size = features1.size(0)
        
        # Compute similarity matrix
        # Concatenate features from both sets
        all_features = torch.cat([features1, features2], dim=0)  # [2*batch_size, feature_dim]
        
        # Compute pairwise similarity
        similarity_matrix = torch.mm(all_features, all_features.t()) / self.temperature
        
        # Create labels for positive pairs
        # Positive pairs are (i, i+batch_size) for i in [0, batch_size)
        positive_indices = torch.arange(batch_size, dtype=torch.long, device=features1.device)
        
        # Create mask for positive pairs
        mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=features1.device)
        for i in range(batch_size):
            mask[i, i + batch_size] = True
            mask[i + batch_size, i] = True
        
        # Mask out self-similarity
        mask_diag = torch.eye(2 * batch_size, dtype=torch.bool, device=features1.device)
        similarity_matrix.masked_fill_(mask_diag, float('-inf'))
        
        # Compute loss for each anchor
        losses = []
        for i in range(2 * batch_size):
            # Get positive similarity
            positive_sim = similarity_matrix[i][mask[i]]
            
            # Get all similarities (excluding self)
            all_sim = similarity_matrix[i]
            
            # Compute log-softmax
            log_prob = F.log_softmax(all_sim, dim=0)
            
            # Loss is negative log probability of positive pair
            positive_log_prob = log_prob[mask[i]]
            loss = -positive_log_prob.mean()
            losses.append(loss)
        
        # Average loss
        total_loss = torch.stack(losses).mean()
        
        return total_loss
    
    def compute_invariance_regularization(self, features, transformed_features):
        """
        Compute a simple invariance regularization term.
        
        Encourages features to be similar after invariance transformations.
        
        Args:
            features: Original features [batch_size, feature_dim]
            transformed_features: Transformed features [batch_size, feature_dim]
        
        Returns:
            Regularization loss (MSE between normalized features)
        """
        # Normalize both
        features = F.normalize(features, p=2, dim=1)
        transformed_features = F.normalize(transformed_features, p=2, dim=1)
        
        # Compute cosine similarity and convert to loss
        similarity = F.cosine_similarity(features, transformed_features, dim=1)
        
        # Loss is 1 - similarity (want to maximize similarity = minimize this)
        loss = (1 - similarity).mean()
        
        return loss
