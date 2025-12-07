"""
Enhanced Optimal Few-Shot Model

This module integrates all invariance modules (task-adaptive, multi-scale,
feature augmentation, medical invariance) with the base OptimalFewShotModel
to improve performance across different datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# Use relative imports for better package structure
import sys
import os

# Get parent directory for imports if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from methods.optimal_few_shot import (
    OptimalFewShotModel, OptimizedConv4, LightweightCosineTransformer,
    DynamicVICRegularizer, EpisodeAdaptiveLambda
)
from modules.task_invariance import TaskAdaptiveInvariance, MultiScaleInvariance
from modules.feature_augmentation import FeatureLevelAugmentation, PrototypicalRefinement
from modules.medical_invariance import MedicalImageInvariance, ContrastiveInvarianceLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EnhancedOptimalFewShot(OptimalFewShotModel):
    """
    Enhanced Optimal Few-Shot Model with Invariance Modules
    
    Extends the base OptimalFewShotModel with additional invariance mechanisms:
    - Task-adaptive invariance
    - Multi-scale invariance (for fine-grained recognition)
    - Feature-level augmentation (for 1-shot robustness)
    - Prototypical refinement
    - Medical image invariance (domain-specific)
    
    Args:
        model_func: Function that returns the backbone network
        n_way: Number of classes in each episode
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        feature_dim: Dimension of transformer features
        n_heads: Number of attention heads in transformer
        dropout: Dropout rate
        num_datasets: Number of different datasets for adaptive lambda
        dataset: Dataset name for configuration
        use_task_invariance: Whether to use task-adaptive invariance
        use_multi_scale: Whether to use multi-scale invariance
        use_feature_augmentation: Whether to use feature augmentation
        use_prototype_refinement: Whether to use prototype refinement
        domain: Domain type ('general', 'medical', 'fine_grained')
        use_focal_loss: Whether to use focal loss
        label_smoothing: Label smoothing factor
    """
    
    def __init__(self, model_func, n_way, k_shot, n_query,
                 feature_dim=64, n_heads=4, dropout=0.1,
                 num_datasets=5, dataset='miniImagenet',
                 use_task_invariance=True,
                 use_multi_scale=False,
                 use_feature_augmentation=True,
                 use_prototype_refinement=False,
                 domain='general',
                 use_focal_loss=False,
                 label_smoothing=0.1):
        
        # Initialize base model
        super(EnhancedOptimalFewShot, self).__init__(
            model_func=model_func,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            feature_dim=feature_dim,
            n_heads=n_heads,
            dropout=dropout,
            num_datasets=num_datasets,
            dataset=dataset,
            use_focal_loss=use_focal_loss,
            label_smoothing=label_smoothing
        )
        
        # Store configuration
        self.use_task_invariance = use_task_invariance
        self.use_multi_scale = use_multi_scale
        self.use_feature_augmentation = use_feature_augmentation
        self.use_prototype_refinement = use_prototype_refinement
        self.domain = domain
        
        # Initialize invariance modules based on configuration
        if self.use_task_invariance:
            self.task_invariance = TaskAdaptiveInvariance(
                feature_dim=feature_dim,
                num_invariance_types=4,
                dropout_rate=dropout
            )
        
        if self.use_multi_scale:
            self.multi_scale = MultiScaleInvariance(
                feature_dim=feature_dim,
                num_scales=3,
                dropout_rate=dropout
            )
        
        if self.use_feature_augmentation:
            self.feature_augmentation = FeatureLevelAugmentation(
                feature_dim=feature_dim,
                num_augmentations=5,
                dropout_rate=dropout
            )
        
        if self.use_prototype_refinement:
            self.prototype_refinement = PrototypicalRefinement(
                feature_dim=feature_dim,
                num_iterations=3,
                dropout_rate=dropout
            )
        
        # Domain-specific modules
        if self.domain == 'medical':
            self.medical_invariance = MedicalImageInvariance(
                feature_dim=feature_dim,
                dropout_rate=dropout
            )
            self.contrastive_loss = ContrastiveInvarianceLoss(temperature=0.07)
    
    def get_parameter_groups(self, lr_backbone_multiplier=0.1):
        """
        Get parameter groups for differential learning rates.
        
        This method provides a robust way to separate backbone parameters
        from invariance module parameters.
        
        Args:
            lr_backbone_multiplier: Multiplier for backbone learning rate
        
        Returns:
            List of parameter groups compatible with PyTorch optimizers
        """
        backbone_params = []
        invariance_params = []
        
        # Explicitly identify backbone parameters
        if hasattr(self, 'feature'):
            backbone_params.extend(self.feature.parameters())
        
        # Projection layer (part of feature processing)
        if hasattr(self, 'projection'):
            backbone_params.extend(self.projection.parameters())
        
        # All other parameters are invariance-related
        backbone_param_ids = {id(p) for p in backbone_params}
        for param in self.parameters():
            if id(param) not in backbone_param_ids:
                invariance_params.append(param)
        
        return [
            {'params': backbone_params, 'lr_multiplier': lr_backbone_multiplier},
            {'params': invariance_params, 'lr_multiplier': 1.0}
        ]
        
    def _apply_invariance_modules(self, features, is_training=True):
        """
        Apply configured invariance modules to features.
        
        Args:
            features: Input features [batch_size, feature_dim]
            is_training: Whether in training mode
        
        Returns:
            Enhanced features [batch_size, feature_dim]
        """
        enhanced = features
        
        # Apply task-adaptive invariance
        if self.use_task_invariance:
            enhanced = self.task_invariance(enhanced)
        
        # Apply multi-scale invariance (for fine-grained recognition)
        if self.use_multi_scale:
            enhanced = self.multi_scale(enhanced)
        
        # Apply feature augmentation (helps 1-shot scenarios)
        if self.use_feature_augmentation:
            enhanced = self.feature_augmentation(enhanced, is_training=is_training)
        
        # Apply domain-specific invariance
        if self.domain == 'medical' and hasattr(self, 'medical_invariance'):
            enhanced = self.medical_invariance(enhanced)
        
        return enhanced
    
    def _set_forward_full(self, x, is_feature=False):
        """
        Internal forward pass with invariance modules.
        
        Overrides base class method to integrate invariance processing.
        """
        z_support, z_query = self.parse_feature(x, is_feature)
        
        # Reshape for transformer
        N_support = z_support.size(0) * z_support.size(1)
        N_query = z_query.size(0) * z_query.size(1)
        
        z_support = z_support.contiguous().reshape(N_support, -1)
        z_query = z_query.contiguous().reshape(N_query, -1)
        
        # Extract features through backbone
        support_features = z_support
        query_features = z_query
        
        # Project to transformer dimension
        support_features = self.projection(support_features)
        query_features = self.projection(query_features)
        
        # Apply invariance modules to features BEFORE transformer
        is_training = self.training
        support_features = self._apply_invariance_modules(support_features, is_training)
        query_features = self._apply_invariance_modules(query_features, is_training)
        
        # Transformer with gradient checkpointing
        # Use try-except for PyTorch version compatibility
        all_features = torch.cat([support_features, query_features], dim=0).unsqueeze(0)
        try:
            # PyTorch 2.0+ with use_reentrant parameter
            all_features = torch.utils.checkpoint.checkpoint(
                self.transformer, all_features, use_reentrant=False
            ).squeeze(0)
        except TypeError:
            # Fallback for older PyTorch versions
            all_features = torch.utils.checkpoint.checkpoint(
                self.transformer, all_features
            ).squeeze(0)
        
        support_features = all_features[:N_support]
        query_features = all_features[N_support:]
        
        # Compute prototypes
        support_features_per_way = support_features.reshape(self.n_way, self.k_shot, -1)
        prototypes = support_features_per_way.mean(dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Apply prototype refinement if enabled
        if self.use_prototype_refinement and hasattr(self, 'prototype_refinement'):
            prototypes = self.prototype_refinement(
                prototypes, query_features, support_features
            )
        
        # Classification logits
        query_norm = F.normalize(query_features, p=2, dim=1)
        proto_norm = F.normalize(prototypes, p=2, dim=1)
        logits = torch.mm(query_norm, proto_norm.t()) * self.temperature
        
        return logits, prototypes, support_features, query_features
    
    def set_forward_loss(self, x):
        """
        Forward pass with loss computation including invariance losses.
        """
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))
        
        logits, prototypes, support_features, query_features = self._set_forward_full(x)
        
        # Get dataset ID
        dataset_id = self.dataset_id_map.get(self.current_dataset, 0)
        
        # Adaptive lambda
        lambda_var, lambda_cov = self.lambda_predictor(
            prototypes, support_features, query_features, dataset_id
        )
        
        # VIC loss
        vic_loss, vic_info = self.vic(
            prototypes, support_features, lambda_var, lambda_cov
        )
        
        # Classification loss
        if self.use_focal_loss:
            ce_loss = self.focal_loss(logits, target)
        else:
            ce_loss = self.loss_fn(logits, target)
        
        # Total loss
        total_loss = ce_loss + vic_loss
        
        # Add contrastive invariance loss for medical domain
        if self.domain == 'medical' and hasattr(self, 'contrastive_loss'):
            # Compute invariance regularization between original and enhanced features
            # This is a simplified version - in practice, you'd use augmented pairs
            invariance_reg = self.contrastive_loss.compute_invariance_regularization(
                support_features, query_features
            )
            total_loss = total_loss + 0.01 * invariance_reg
        
        # Calculate accuracy
        predict = torch.argmax(logits, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        
        return acc, total_loss
