"""
Experiment Configuration Classes for Unified Runner

This module defines configuration classes for:
1. Main experiment configuration
2. Ablation study configurations
3. VIC component settings
4. Run mode specifications
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict


class RunMode(Enum):
    """Available run modes for the experiment runner"""
    ALL = "all"
    TRAIN_TEST = "train_test"
    ABLATION = "ablation"
    QUALITATIVE = "qualitative"
    FEATURE_ANALYSIS = "feature_analysis"
    MCNEMAR = "mcnemar"


@dataclass
class VICComponents:
    """Configuration for VIC (Variance-Invariance-Covariance) components"""
    invariance: bool = True
    covariance: bool = True
    variance: bool = True
    dynamic_weight: bool = True
    
    def __str__(self):
        """String representation for logging"""
        components = []
        if self.invariance:
            components.append("I")
        if self.covariance:
            components.append("C")
        if self.variance:
            components.append("V")
        if self.dynamic_weight:
            components.append("D")
        return "".join(components) if components else "Baseline"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'invariance': self.invariance,
            'covariance': self.covariance,
            'variance': self.variance,
            'dynamic_weight': self.dynamic_weight
        }


@dataclass
class AblationExperimentConfig:
    """Configuration for a single ablation experiment"""
    name: str
    vic_components: VICComponents
    description: str
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'vic_components': self.vic_components.to_dict(),
            'description': self.description
        }


# Pre-defined ablation experiments as per specification
ABLATION_EXPERIMENTS = {
    'E1': AblationExperimentConfig(
        name='E1_Full',
        vic_components=VICComponents(
            invariance=True,
            covariance=True,
            variance=True,
            dynamic_weight=True
        ),
        description='Full Dynamic VIC model'
    ),
    'E2': AblationExperimentConfig(
        name='E2_InvDyn',
        vic_components=VICComponents(
            invariance=True,
            covariance=False,
            variance=False,
            dynamic_weight=True
        ),
        description='Only invariance + dynamic weight'
    ),
    'E3': AblationExperimentConfig(
        name='E3_InvCovDyn',
        vic_components=VICComponents(
            invariance=True,
            covariance=True,
            variance=False,
            dynamic_weight=True
        ),
        description='Invariance + covariance + dynamic'
    ),
    'E4': AblationExperimentConfig(
        name='E4_InvVarDyn',
        vic_components=VICComponents(
            invariance=True,
            covariance=False,
            variance=True,
            dynamic_weight=True
        ),
        description='Invariance + variance + dynamic'
    ),
    'E5': AblationExperimentConfig(
        name='E5_FullNoD',
        vic_components=VICComponents(
            invariance=True,
            covariance=True,
            variance=True,
            dynamic_weight=False
        ),
        description='Full VIC without dynamic weight'
    ),
    'E6': AblationExperimentConfig(
        name='E6_Baseline',
        vic_components=VICComponents(
            invariance=False,
            covariance=False,
            variance=False,
            dynamic_weight=False
        ),
        description='Baseline cosine similarity (comparison method)'
    ),
    'E7': AblationExperimentConfig(
        name='E7_CovDyn',
        vic_components=VICComponents(
            invariance=False,
            covariance=True,
            variance=False,
            dynamic_weight=True
        ),
        description='Only covariance + dynamic'
    ),
    'E8': AblationExperimentConfig(
        name='E8_VarDyn',
        vic_components=VICComponents(
            invariance=False,
            covariance=False,
            variance=True,
            dynamic_weight=True
        ),
        description='Only variance + dynamic'
    ),
}


@dataclass
class ExperimentConfig:
    """Main configuration for unified experiment runner"""
    # Dataset and model settings
    dataset: str = 'miniImagenet'
    backbone: str = 'Conv4'
    n_way: int = 5
    k_shot: int = 1
    n_query: int = 16
    
    # Training settings
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimization: str = 'AdamW'
    
    # Testing settings
    test_iter: int = 600
    
    # Run mode
    run_mode: RunMode = RunMode.ALL
    
    # Output settings
    output_dir: str = './results'
    seed: int = 4040
    
    # Ablation study settings
    ablation_experiments: List[str] = field(default_factory=lambda: list(ABLATION_EXPERIMENTS.keys()))
    
    # Feature analysis settings
    max_episodes_for_visualization: int = 100
    
    # Model checkpoint settings
    baseline_checkpoint: Optional[str] = None
    proposed_checkpoint: Optional[str] = None
    
    # Hardware settings
    device: str = 'cuda'
    num_workers: int = 4
    
    # Logging settings
    verbose: bool = True
    save_visualizations: bool = True
    
    def get_experiment_name(self):
        """Generate experiment name for directory structure"""
        return f"{self.dataset}_{self.backbone}_{self.n_way}w{self.k_shot}s"
    
    def get_output_paths(self):
        """Generate output directory structure"""
        base_path = f"{self.output_dir}/{self.get_experiment_name()}"
        return {
            'base': base_path,
            'quantitative': f"{base_path}/quantitative",
            'qualitative': f"{base_path}/qualitative",
            'ablation': f"{base_path}/ablation",
            'mcnemar': f"{base_path}/mcnemar",
            'feature_analysis': f"{base_path}/feature_analysis"
        }
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'dataset': self.dataset,
            'backbone': self.backbone,
            'n_way': self.n_way,
            'k_shot': self.k_shot,
            'n_query': self.n_query,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimization': self.optimization,
            'test_iter': self.test_iter,
            'run_mode': self.run_mode.value,
            'output_dir': self.output_dir,
            'seed': self.seed,
            'device': self.device
        }
