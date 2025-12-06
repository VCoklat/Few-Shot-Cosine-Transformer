"""
Invariance modules for enhanced few-shot learning.

This package contains modules that implement various invariance mechanisms
to improve few-shot learning performance, especially in 1-shot scenarios.
"""

from .task_invariance import TaskAdaptiveInvariance, MultiScaleInvariance
from .feature_augmentation import FeatureLevelAugmentation, PrototypicalRefinement
from .medical_invariance import MedicalImageInvariance, ContrastiveInvarianceLoss

__all__ = [
    'TaskAdaptiveInvariance',
    'MultiScaleInvariance',
    'FeatureLevelAugmentation',
    'PrototypicalRefinement',
    'MedicalImageInvariance',
    'ContrastiveInvarianceLoss',
]
