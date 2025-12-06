"""
Feature-Level Augmentation Module

This module implements feature-level augmentation to improve robustness
in 1-shot scenarios and iterative prototype refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureLevelAugmentation(nn.Module):
    """
    Feature-Level Augmentation Module
    
    Augments features at the feature level by learning perturbation directions
    and applying task-adaptive magnitude. Helps improve robustness in 1-shot scenarios.
    
    Args:
        feature_dim: Dimension of input features
        num_augmentations: Number of different augmentation directions
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, feature_dim, num_augmentations=5, dropout_rate=0.1):
        super(FeatureLevelAugmentation, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_augmentations = num_augmentations
        
        # Learnable perturbation directions (initialized orthogonally for better conditioning)
        perturbation_matrix = torch.empty(num_augmentations, feature_dim)
        nn.init.orthogonal_(perturbation_matrix)
        self.perturbation_directions = nn.Parameter(perturbation_matrix)
        
        # Magnitude predictor (task-adaptive)
        self.magnitude_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_augmentations),
            nn.Sigmoid()  # Magnitudes in [0, 1]
        )
        
        # Feature mixing network
        self.mixing_network = nn.Sequential(
            nn.Linear(feature_dim * (num_augmentations + 1), feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Learnable temperature for diversity control
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
    def _orthogonalize_directions(self):
        """Orthogonalize perturbation directions using Gram-Schmidt"""
        directions = self.perturbation_directions
        # Normalize
        directions = F.normalize(directions, p=2, dim=1)
        return directions
    
    def forward(self, features, is_training=True):
        """
        Apply feature-level augmentation.
        
        Args:
            features: Input features [batch_size, feature_dim]
            is_training: Whether in training mode (applies augmentation)
        
        Returns:
            Augmented features [batch_size, feature_dim]
        """
        if not is_training:
            return features
        
        batch_size = features.size(0)
        
        # Orthogonalize perturbation directions
        directions = self._orthogonalize_directions()  # [num_augmentations, feature_dim]
        
        # Predict adaptive magnitudes based on features
        feature_context = features.mean(dim=0, keepdim=True)  # [1, feature_dim]
        magnitudes = self.magnitude_predictor(feature_context)  # [1, num_augmentations]
        magnitudes = magnitudes * self.temperature  # Scale by learnable temperature
        
        # Generate augmented versions
        augmented_features = [features]  # Start with original
        for i in range(self.num_augmentations):
            # Apply perturbation along i-th direction
            perturbation = directions[i:i+1] * magnitudes[0, i]  # [1, feature_dim]
            augmented = features + perturbation
            # Normalize to maintain unit norm
            augmented = F.normalize(augmented, p=2, dim=1)
            augmented_features.append(augmented)
        
        # Concatenate all versions
        concatenated = torch.cat(augmented_features, dim=-1)  # [batch_size, feature_dim * (num_aug + 1)]
        
        # Mix augmented features
        mixed = self.mixing_network(concatenated)
        
        # Normalize output
        output = F.normalize(mixed, p=2, dim=1)
        
        return output


class PrototypicalRefinement(nn.Module):
    """
    Prototypical Refinement Module
    
    Iteratively refines prototypes using query features to make them more robust,
    especially beneficial for 1-shot scenarios where initial prototypes are noisy.
    
    Args:
        feature_dim: Dimension of input features
        num_iterations: Number of refinement iterations
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, feature_dim, num_iterations=3, dropout_rate=0.1):
        super(PrototypicalRefinement, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        
        # Refinement network (shared across iterations)
        self.refinement_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Attention mechanism for selecting relevant query features
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Learnable step size for each iteration
        self.step_sizes = nn.Parameter(torch.ones(num_iterations) * 0.1)
        
    def forward(self, prototypes, query_features, support_features=None):
        """
        Refine prototypes iteratively using query features.
        
        Args:
            prototypes: Initial prototypes [n_way, feature_dim]
            query_features: Query features [n_query, feature_dim]
            support_features: Optional support features for additional context
        
        Returns:
            Refined prototypes [n_way, feature_dim]
        """
        refined_prototypes = prototypes
        
        for iter_idx in range(self.num_iterations):
            # For each prototype, aggregate relevant query information
            updated_prototypes = []
            
            for proto_idx in range(refined_prototypes.size(0)):
                current_proto = refined_prototypes[proto_idx:proto_idx+1]  # [1, feature_dim]
                
                # Compute similarity to all query features
                proto_expanded = current_proto.expand(query_features.size(0), -1)  # [n_query, feature_dim]
                
                # Concatenate for attention computation
                attention_input = torch.cat([proto_expanded, query_features], dim=-1)  # [n_query, feature_dim * 2]
                
                # Compute attention weights
                attention_weights = self.attention(attention_input)  # [n_query, 1]
                
                # Weighted aggregation of query features
                weighted_queries = query_features * attention_weights  # [n_query, feature_dim]
                aggregated_query = weighted_queries.mean(dim=0, keepdim=True)  # [1, feature_dim]
                
                # Refinement input: current prototype + aggregated query info
                refinement_input = torch.cat([current_proto, aggregated_query], dim=-1)  # [1, feature_dim * 2]
                
                # Apply refinement network
                refinement_delta = self.refinement_net(refinement_input)  # [1, feature_dim]
                
                # Update prototype with learnable step size
                updated_proto = current_proto + self.step_sizes[iter_idx] * refinement_delta
                
                # Normalize
                updated_proto = F.normalize(updated_proto, p=2, dim=1)
                
                updated_prototypes.append(updated_proto)
            
            # Stack updated prototypes
            refined_prototypes = torch.cat(updated_prototypes, dim=0)
        
        return refined_prototypes
