#!/usr/bin/env python3
"""
Unified Experiment Runner for Few-Shot Cosine Transformer

This script provides a comprehensive framework for running experiments including:
- Training and testing with quantitative measurements
- Qualitative analysis (t-SNE, confusion matrices, attention maps)
- Ablation studies with dynamic VIC components
- McNemar's significance testing
- Feature collapse analysis

Usage:
    python run_experiments.py --dataset miniImagenet --backbone Conv4 --run_mode all
    python run_experiments.py --dataset CUB --backbone ResNet18 --run_mode ablation
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backbone
import configs as data_configs
from configs.experiment_config import (
    ExperimentConfig,
    AblationExperimentConfig,
    VICComponents,
    RunMode,
    ABLATION_EXPERIMENTS
)
from data.datamgr import SetDataManager
from io_utils import model_dict
from methods.transformer import FewShotTransformer
from methods.optimal_few_shot import OptimalFewShot

# Set up logging first (before other imports that may use logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import utility modules with error handling
try:
    import eval_utils
    EVAL_UTILS_AVAILABLE = True
except ImportError:
    EVAL_UTILS_AVAILABLE = False
    logger.warning("eval_utils not available. Some features may be limited.")

try:
    from feature_analysis import (
        compute_confidence_interval,
        detect_feature_collapse,
        compute_feature_utilization,
        comprehensive_feature_analysis
    )
    FEATURE_ANALYSIS_AVAILABLE = True
except ImportError:
    FEATURE_ANALYSIS_AVAILABLE = False
    # Define fallback for compute_confidence_interval
    def compute_confidence_interval(accuracies, confidence=0.95):
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        n = len(accuracies)
        z_score = 1.96 if confidence == 0.95 else 2.576
        margin = z_score * std / np.sqrt(n)
        return mean, mean - margin, mean + margin
    logger.warning("feature_analysis not available. Using fallback implementation.")

try:
    from ablation_study import (
        mcnemar_test,
        mcnemar_test_multiple,
        compute_contingency_table,
        format_contingency_table
    )
    ABLATION_STUDY_AVAILABLE = True
except ImportError:
    ABLATION_STUDY_AVAILABLE = False
    logger.warning("ablation_study not available. McNemar's test will not be available.")

# Global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def setup_output_directories(config: ExperimentConfig):
    """Create output directory structure"""
    paths = config.get_output_paths()
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")
    return paths


def filter_results_for_json(results: Dict, exclude_keys: Optional[List[str]] = None) -> Dict:
    """
    Filter results dictionary for JSON serialization by excluding large arrays.
    
    Args:
        results: Results dictionary to filter
        exclude_keys: List of keys to exclude (default: features, predictions, true_labels)
    
    Returns:
        Filtered copy of results dictionary
    """
    if exclude_keys is None:
        exclude_keys = ['features', 'predictions', 'true_labels']
    
    return {k: v for k, v in results.items() if k not in exclude_keys}


def safe_plot_save(output_path: str, dpi: int = 150, show: bool = False):
    """
    Safely save and close a matplotlib plot with proper exception handling.
    
    Args:
        output_path: Path to save the plot
        dpi: DPI for the saved image
        show: Whether to display the plot interactively before closing
    """
    try:
        plt.savefig(output_path, dpi=dpi)
        logger.info(f"  Saved: {output_path}")
        if show:
            plt.show()
    except Exception as e:
        logger.error(f"  Failed to save plot to {output_path}: {e}")
    finally:
        plt.close()


def run_mcnemar_comparison(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray, 
                          name_a: str, name_b: str) -> Optional[Dict]:
    """
    Run and display McNemar's test comparison between two models.
    
    Args:
        preds_a: Predictions from model A
        preds_b: Predictions from model B
        labels: True labels
        name_a: Name of model A
        name_b: Name of model B
    
    Returns:
        McNemar test result dictionary or None if test not available
    """
    if not ABLATION_STUDY_AVAILABLE:
        logger.warning("McNemar's test not available - ablation_study module not imported")
        return None
    
    try:
        # Validate inputs
        if len(preds_a) != len(preds_b) or len(preds_a) != len(labels):
            logger.error(f"Prediction and label arrays must have the same length: "
                        f"preds_a={len(preds_a)}, preds_b={len(preds_b)}, labels={len(labels)}")
            return None
        
        result = mcnemar_test(preds_a, preds_b, labels)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"McNEMAR'S TEST: {name_a} vs {name_b}")
        logger.info(f"{'='*60}")
        logger.info(f"  p-value: {result['p_value']:.6f}")
        logger.info(f"  Significant at 0.05: {result['significant_at_0.05']}")
        logger.info(f"  Effect: {result['effect_description']}")
        logger.info(f"  Discordant pairs: {result['discordant_pairs']}")
        
        # Display contingency table summary
        n00, n01, n10, n11 = result['contingency_table']
        logger.info(f"  Contingency table:")
        logger.info(f"    Both correct: {n11}")
        logger.info(f"    Both wrong: {n00}")
        logger.info(f"    {name_a} correct, {name_b} wrong: {n10}")
        logger.info(f"    {name_a} wrong, {name_b} correct: {n01}")
        logger.info(f"{'='*60}\n")
        
        return result
    except Exception as e:
        logger.error(f"Error running McNemar's test: {e}")
        return None


def get_data_loaders(config: ExperimentConfig):
    """Create data loaders for training/validation/testing"""
    logger.info(f"Setting up data loaders for {config.dataset}")
    
    # Get dataset path
    if config.dataset not in data_configs.data_dir:
        raise ValueError(f"Dataset {config.dataset} not found in configs")
    
    dataset_path = data_configs.data_dir[config.dataset]
    
    # Determine image size based on dataset
    if config.dataset in ['Omniglot', 'cross_char']:
        image_size = 28
    elif config.dataset == 'CIFAR':
        image_size = 32
    else:
        image_size = 84
    
    # Create data managers
    train_few_shot_params = dict(
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.n_query,
        n_episode=200
    )
    
    test_few_shot_params = dict(
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.n_query,
        n_episode=config.test_iter
    )
    
    # Training loader
    train_datamgr = SetDataManager(
        image_size,
        **train_few_shot_params
    )
    train_loader = train_datamgr.get_data_loader(
        os.path.join(dataset_path, 'base.json'),
        aug=False
    )
    
    # Validation loader
    val_datamgr = SetDataManager(
        image_size,
        **train_few_shot_params
    )
    val_loader = val_datamgr.get_data_loader(
        os.path.join(dataset_path, 'val.json'),
        aug=False
    )
    
    # Test loader
    test_datamgr = SetDataManager(
        image_size,
        **test_few_shot_params
    )
    test_loader = test_datamgr.get_data_loader(
        os.path.join(dataset_path, 'novel.json'),
        aug=False
    )
    
    logger.info(f"Data loaders created successfully")
    logger.info(f"  Image size: {image_size}")
    logger.info(f"  Training episodes: {len(train_loader)}")
    logger.info(f"  Validation episodes: {len(val_loader)}")
    logger.info(f"  Test episodes: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def create_model(model_type: str, config: ExperimentConfig, vic_components: Optional[VICComponents] = None):
    """
    Create a model instance based on configuration
    
    Args:
        model_type: 'baseline' or 'proposed'
        config: Experiment configuration
        vic_components: Optional VIC components configuration for ablation studies
    """
    logger.info(f"Creating {model_type} model")
    
    # Get backbone
    if config.backbone not in model_dict:
        raise ValueError(f"Backbone {config.backbone} not supported")
    
    model_func = model_dict[config.backbone]
    
    if model_type == 'baseline' or (vic_components and vic_components.is_baseline()):
        # Create baseline FewShotTransformer
        model = FewShotTransformer(
            model_func,
            n_way=config.n_way,
            k_shot=config.k_shot,
            n_query=config.n_query,
            variant='cosine',
            depth=1,
            heads=4,
            dim_head=64,
            mlp_dim=512,
            dataset=config.dataset,
            feti=0,
            flatten=False
        )
        logger.info(f"  Model: FewShotTransformer (baseline)")
    
    elif model_type == 'proposed':
        # Create OptimalFewShot model
        model = OptimalFewShot(
            model_func,
            n_way=config.n_way,
            k_shot=config.k_shot,
            n_query=config.n_query,
            dataset=config.dataset,
            use_vic=True,
            vic_lambda_inv=1.0,
            vic_lambda_cov=1.0,
            vic_lambda_var=1.0,
            use_dynamic_lambda=True
        )
        logger.info(f"  Model: OptimalFewShot (proposed)")
    
    elif vic_components:
        # Create model with specific VIC components for ablation
        model = OptimalFewShot(
            model_func,
            n_way=config.n_way,
            k_shot=config.k_shot,
            n_query=config.n_query,
            dataset=config.dataset,
            use_vic=(vic_components.invariance or vic_components.covariance or vic_components.variance),
            vic_lambda_inv=1.0 if vic_components.invariance else 0.0,
            vic_lambda_cov=1.0 if vic_components.covariance else 0.0,
            vic_lambda_var=1.0 if vic_components.variance else 0.0,
            use_dynamic_lambda=vic_components.dynamic_weight
        )
        logger.info(f"  Model: OptimalFewShot with VIC={vic_components}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Parameters: {param_count:.2f}M")
    
    return model


def train_model(model, train_loader, val_loader, config: ExperimentConfig, output_dir: str):
    """Train a model and save the best checkpoint"""
    logger.info("="*80)
    logger.info("TRAINING MODEL")
    logger.info("="*80)
    
    # Setup optimizer
    if config.optimization == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimization == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimization == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f'Unknown optimization: {config.optimization}')
    
    best_acc = 0.0
    best_epoch = 0
    training_start_time = time.time()
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        # Training loop
        train_acc_list = []
        train_loss_list = []
        
        with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
            for i, (x, _) in enumerate(train_loader):
                optimizer.zero_grad()
                acc, loss = model.set_forward_loss(x)
                loss.backward()
                optimizer.step()
                
                train_acc_list.append(acc)
                train_loss_list.append(loss.item())
                
                pbar.set_postfix({
                    'loss': f"{np.mean(train_loss_list):.4f}",
                    'acc': f"{np.mean(train_acc_list):.4f}"
                })
                pbar.update(1)
        
        epoch_time = time.time() - epoch_start_time
        train_acc = np.mean(train_acc_list)
        train_loss = np.mean(train_loss_list)
        
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} - "
                   f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Time: {epoch_time:.2f}s")
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_acc_list = []
            for x, _ in val_loader:
                acc, _ = model.set_forward_loss(x)
                val_acc_list.append(acc)
            
            val_acc = np.mean(val_acc_list)
            logger.info(f"  Validation Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                checkpoint_path = os.path.join(output_dir, 'best_model.tar')
                torch.save({
                    'epoch': epoch,
                    'state': model.state_dict(),
                    'accuracy': best_acc
                }, checkpoint_path)
                logger.info(f"  ✓ New best model saved (acc: {best_acc:.4f})")
    
    training_time = time.time() - training_start_time
    
    logger.info(f"\nTraining completed in {training_time:.2f}s")
    logger.info(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    
    return {
        'training_time': training_time,
        'best_val_acc': best_acc,
        'best_epoch': best_epoch
    }


def test_model(model, test_loader, config: ExperimentConfig, extract_features: bool = False):
    """Test a model and return comprehensive metrics"""
    logger.info("="*80)
    logger.info("TESTING MODEL")
    logger.info("="*80)
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    episode_accs = []
    inference_times = []
    all_features = [] if extract_features else None
    
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_loader), desc="Testing") as pbar:
            for x, _ in test_loader:
                start_time = time.time()
                
                scores = model.set_forward(x)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                preds = scores.argmax(1).cpu().numpy()
                n_query_per_way = len(preds) // config.n_way
                labels = np.repeat(np.arange(config.n_way), n_query_per_way)
                
                all_preds.append(preds)
                all_labels.append(labels)
                all_scores.append(scores.cpu().numpy())
                
                episode_acc = np.mean(preds == labels)
                episode_accs.append(episode_acc)
                
                # Extract features if requested
                if extract_features:
                    if hasattr(model, 'parse_feature'):
                        z_support, z_query = model.parse_feature(x, is_feature=False)
                        features = z_query.reshape(-1, z_query.size(-1)).cpu().numpy()
                        all_features.append(features)
                
                pbar.set_postfix({'acc': f"{np.mean(episode_accs):.4f}"})
                pbar.update(1)
    
    # Aggregate results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Confidence interval
    mean_acc, lower_ci, upper_ci = compute_confidence_interval(np.array(episode_accs))
    
    # Parameter count
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Confusion matrix
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': float(accuracy),
        'std': float(np.std(episode_accs)),
        'precision': float(precision),
        'recall': float(recall),
        'f1_macro': float(f1),
        'confidence_interval_95': {
            'mean': float(mean_acc),
            'lower': float(lower_ci),
            'upper': float(upper_ci),
            'margin': float(mean_acc - lower_ci)
        },
        'param_count': float(param_count),
        'avg_inference_time': float(np.mean(inference_times)),
        'confusion_matrix': conf_mat.tolist(),
        'predictions': all_preds.tolist(),
        'true_labels': all_labels.tolist()
    }
    
    # Add feature data if extracted
    if extract_features and all_features:
        results['features'] = np.concatenate(all_features, axis=0)
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {accuracy:.4f} ± {np.std(episode_accs):.4f}")
    logger.info(f"  95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 (macro): {f1:.4f}")
    logger.info(f"  Parameters: {param_count:.2f}M")
    logger.info(f"  Avg inference time: {np.mean(inference_times)*1000:.2f}ms")
    
    return results


def run_train_test(config: ExperimentConfig, output_paths: Dict[str, str]):
    """Run training and testing for baseline and proposed models"""
    logger.info("\n" + "="*80)
    logger.info("PHASE: TRAIN AND TEST")
    logger.info("="*80)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    results = {}
    
    # ========== Baseline Model ==========
    if config.baseline_checkpoint and os.path.exists(config.baseline_checkpoint):
        logger.info("\nLoading baseline model from checkpoint...")
        baseline_model = create_model('baseline', config)
        checkpoint = torch.load(config.baseline_checkpoint, weights_only=False)
        baseline_model.load_state_dict(checkpoint['state'])
        baseline_train_results = {'loaded_from_checkpoint': config.baseline_checkpoint}
    else:
        logger.info("\n>>> Training BASELINE model (fsct_cosine)")
        baseline_model = create_model('baseline', config)
        baseline_train_results = train_model(
            baseline_model,
            train_loader,
            val_loader,
            config,
            output_paths['quantitative']
        )
        # Load best model
        checkpoint_path = os.path.join(output_paths['quantitative'], 'best_model.tar')
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        baseline_model.load_state_dict(checkpoint['state'])
    
    logger.info("\n>>> Testing BASELINE model")
    baseline_test_results = test_model(baseline_model, test_loader, config, extract_features=True)
    
    results['baseline'] = {
        'training': baseline_train_results,
        'testing': baseline_test_results
    }
    
    # Save baseline results
    with open(os.path.join(output_paths['quantitative'], 'baseline_results.json'), 'w') as f:
        save_results = filter_results_for_json(baseline_test_results)
        save_results['training'] = baseline_train_results
        json.dump(save_results, f, indent=2)
    
    # ========== Proposed Model ==========
    if config.proposed_checkpoint and os.path.exists(config.proposed_checkpoint):
        logger.info("\nLoading proposed model from checkpoint...")
        proposed_model = create_model('proposed', config)
        checkpoint = torch.load(config.proposed_checkpoint, weights_only=False)
        proposed_model.load_state_dict(checkpoint['state'])
        proposed_train_results = {'loaded_from_checkpoint': config.proposed_checkpoint}
    else:
        logger.info("\n>>> Training PROPOSED model (optimalfewshot)")
        proposed_model = create_model('proposed', config)
        proposed_train_results = train_model(
            proposed_model,
            train_loader,
            val_loader,
            config,
            output_paths['quantitative']
        )
        # Load best model
        checkpoint_path = os.path.join(output_paths['quantitative'], 'best_model.tar')
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        proposed_model.load_state_dict(checkpoint['state'])
    
    logger.info("\n>>> Testing PROPOSED model")
    proposed_test_results = test_model(proposed_model, test_loader, config, extract_features=True)
    
    results['proposed'] = {
        'training': proposed_train_results,
        'testing': proposed_test_results
    }
    
    # Save proposed results
    with open(os.path.join(output_paths['quantitative'], 'proposed_results.json'), 'w') as f:
        save_results = filter_results_for_json(proposed_test_results)
        save_results['training'] = proposed_train_results
        json.dump(save_results, f, indent=2)
    
    # ========== Comparison ==========
    comparison = {
        'baseline_accuracy': baseline_test_results['accuracy'],
        'proposed_accuracy': proposed_test_results['accuracy'],
        'accuracy_improvement': proposed_test_results['accuracy'] - baseline_test_results['accuracy'],
        'baseline_f1': baseline_test_results['f1_macro'],
        'proposed_f1': proposed_test_results['f1_macro'],
        'f1_improvement': proposed_test_results['f1_macro'] - baseline_test_results['f1_macro'],
        'baseline_params': baseline_test_results['param_count'],
        'proposed_params': proposed_test_results['param_count'],
        'param_overhead': proposed_test_results['param_count'] - baseline_test_results['param_count']
    }
    
    results['comparison'] = comparison
    
    # Save comparison
    with open(os.path.join(output_paths['quantitative'], 'comparison_metrics.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"Baseline Accuracy: {comparison['baseline_accuracy']:.4f}")
    logger.info(f"Proposed Accuracy: {comparison['proposed_accuracy']:.4f}")
    logger.info(f"Improvement: {comparison['accuracy_improvement']:.4f} ({comparison['accuracy_improvement']*100:.2f}%)")
    logger.info(f"\nBaseline F1: {comparison['baseline_f1']:.4f}")
    logger.info(f"Proposed F1: {comparison['proposed_f1']:.4f}")
    logger.info(f"Improvement: {comparison['f1_improvement']:.4f}")
    logger.info(f"\nBaseline Params: {comparison['baseline_params']:.2f}M")
    logger.info(f"Proposed Params: {comparison['proposed_params']:.2f}M")
    logger.info(f"Overhead: {comparison['param_overhead']:.2f}M ({comparison['param_overhead']/comparison['baseline_params']*100:.2f}%)")
    
    # ========== McNemar's Test (if enabled) ==========
    if config.mcnemar_each_test:
        logger.info("\n>>> Running McNemar's significance test")
        
        baseline_preds = np.array(baseline_test_results['predictions'])
        proposed_preds = np.array(proposed_test_results['predictions'])
        baseline_labels = np.array(baseline_test_results['true_labels'])
        proposed_labels = np.array(proposed_test_results['true_labels'])
        
        # Validate that both models were tested on the same data
        if not np.array_equal(baseline_labels, proposed_labels):
            logger.error("Cannot run McNemar's test: baseline and proposed models have different true labels")
        else:
            mcnemar_result = run_mcnemar_comparison(
                baseline_preds,
                proposed_preds,
                baseline_labels,
                'Baseline',
                'Proposed'
            )
            
            if mcnemar_result:
                # Save McNemar result
                with open(os.path.join(output_paths['quantitative'], 'mcnemar_test.json'), 'w') as f:
                    json.dump(mcnemar_result, f, indent=2)
                
                results['mcnemar_test'] = mcnemar_result
    
    return results


def run_qualitative_analysis(config: ExperimentConfig, output_paths: Dict[str, str], 
                            test_results: Optional[Dict] = None):
    """Generate qualitative visualizations"""
    logger.info("\n" + "="*80)
    logger.info("PHASE: QUALITATIVE ANALYSIS")
    logger.info("="*80)
    
    # If we don't have test results, we need to run testing first
    if test_results is None:
        logger.info("Running tests to extract features for visualization...")
        _, _, test_loader = get_data_loaders(config)
        
        baseline_model = create_model('baseline', config)
        proposed_model = create_model('proposed', config)
        
        # Load checkpoints if available
        baseline_checkpoint = os.path.join(output_paths['quantitative'], 'best_model.tar')
        if os.path.exists(baseline_checkpoint):
            checkpoint = torch.load(baseline_checkpoint, weights_only=False)
            baseline_model.load_state_dict(checkpoint['state'])
        
        baseline_results = test_model(baseline_model, test_loader, config, extract_features=True)
        proposed_results = test_model(proposed_model, test_loader, config, extract_features=True)
        
        test_results = {
            'baseline': {'testing': baseline_results},
            'proposed': {'testing': proposed_results}
        }
    
    # Generate confusion matrices
    logger.info("\n>>> Generating confusion matrices")
    
    for model_name in ['baseline', 'proposed']:
        if model_name not in test_results:
            continue
        
        results = test_results[model_name]['testing']
        conf_mat = np.array(results['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(config.n_way), yticklabels=range(config.n_way))
        plt.title(f'Confusion Matrix - {model_name.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        output_path = os.path.join(output_paths['qualitative'], f'confusion_matrix_{model_name}.png')
        safe_plot_save(output_path, show=config.show_plots)
    
    # Generate t-SNE visualization if features are available
    logger.info("\n>>> Generating t-SNE visualization")
    
    try:
        from sklearn.manifold import TSNE
        
        for model_name in ['baseline', 'proposed']:
            if model_name not in test_results:
                continue
            
            results = test_results[model_name]['testing']
            
            if 'features' in results and results['features'] is not None:
                features = np.array(results['features'])
                labels = np.array(results['true_labels'])[:len(features)]
                
                # Sample for t-SNE if too many points
                if len(features) > 1000:
                    indices = np.random.choice(len(features), 1000, replace=False)
                    features = features[indices]
                    labels = labels[indices]
                
                tsne = TSNE(n_components=2, random_state=config.seed, perplexity=30)
                features_2d = tsne.fit_transform(features)
                
                plt.figure(figsize=(12, 10))
                scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.6, s=50)
                plt.colorbar(scatter, label='Class')
                plt.title(f't-SNE Visualization - {model_name.capitalize()}')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.tight_layout()
                
                output_path = os.path.join(output_paths['qualitative'], f'tsne_{model_name}.png')
                safe_plot_save(output_path, show=config.show_plots)
            else:
                logger.warning(f"  No features available for {model_name}")
    
    except ImportError:
        logger.warning("  scikit-learn not available, skipping t-SNE visualization")
    except Exception as e:
        logger.error(f"  Error generating t-SNE: {e}")
    
    logger.info("\nQualitative analysis completed")


def run_ablation_study(config: ExperimentConfig, output_paths: Dict[str, str]):
    """Run ablation study with different VIC component configurations"""
    logger.info("\n" + "="*80)
    logger.info("PHASE: ABLATION STUDY")
    logger.info("="*80)
    
    # Get data loaders
    _, _, test_loader = get_data_loaders(config)
    
    ablation_results = {}
    all_predictions = []
    all_model_names = []
    true_labels = None
    
    # Run each ablation experiment
    for exp_key in config.ablation_experiments:
        if exp_key not in ABLATION_EXPERIMENTS:
            logger.warning(f"Skipping unknown experiment: {exp_key}")
            continue
        
        exp_config = ABLATION_EXPERIMENTS[exp_key]
        logger.info(f"\n>>> Running {exp_config.name}: {exp_config.description}")
        
        # Create model with specific VIC components
        model = create_model('proposed', config, vic_components=exp_config.vic_components)
        
        # For E6 (baseline), use the baseline architecture
        if exp_key == 'E6':
            model = create_model('baseline', config)
        
        # Test the model
        test_results = test_model(model, test_loader, config)
        
        ablation_results[exp_config.name] = {
            'config': exp_config.to_dict(),
            'results': test_results
        }
        
        all_predictions.append(np.array(test_results['predictions']))
        all_model_names.append(exp_config.name)
        
        if true_labels is None:
            true_labels = np.array(test_results['true_labels'])
        
        logger.info(f"  Accuracy: {test_results['accuracy']:.4f} ± {test_results['std']:.4f}")
        
        # ========== McNemar's Test against baseline (if enabled) ==========
        if config.mcnemar_each_test and exp_key != 'E6':
            # Compare against E6 (baseline) if it has been tested
            if 'E6_Baseline' in ablation_results and true_labels is not None:
                logger.info(f"\n>>> Running McNemar's test: {exp_config.name} vs E6_Baseline")
                
                baseline_preds = np.array(ablation_results['E6_Baseline']['results']['predictions'])
                current_preds = np.array(test_results['predictions'])
                
                mcnemar_result = run_mcnemar_comparison(
                    baseline_preds,
                    current_preds,
                    true_labels,
                    'E6_Baseline',
                    exp_config.name
                )
                
                if mcnemar_result:
                    ablation_results[exp_config.name]['mcnemar_vs_baseline'] = mcnemar_result
            elif exp_key != 'E6':
                logger.info(f"  Note: E6_Baseline not yet tested, will compare in final McNemar phase")
    
    # Save ablation results
    ablation_summary = {}
    for name, data in ablation_results.items():
        ablation_summary[name] = {
            'description': data['config']['description'],
            'vic_components': data['config']['vic_components'],
            'accuracy': data['results']['accuracy'],
            'std': data['results']['std'],
            'f1_macro': data['results']['f1_macro'],
            'confidence_interval': data['results']['confidence_interval_95']
        }
    
    with open(os.path.join(output_paths['ablation'], 'ablation_results.json'), 'w') as f:
        json.dump(ablation_summary, f, indent=2)
    
    # Generate ablation comparison plot
    logger.info("\n>>> Generating ablation comparison plot")
    
    names = list(ablation_summary.keys())
    accuracies = [ablation_summary[n]['accuracy'] for n in names]
    stds = [ablation_summary[n]['std'] for n in names]
    
    plt.figure(figsize=(14, 8))
    x_pos = np.arange(len(names))
    plt.bar(x_pos, accuracies, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Ablation Study: Component Contribution to Accuracy', fontsize=14)
    plt.xticks(x_pos, names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_paths['ablation'], 'ablation_comparison.png')
    safe_plot_save(output_path, show=config.show_plots)
    
    # Analyze component importance
    logger.info("\n>>> Analyzing component importance")
    
    baseline_acc = ablation_summary['E6_Baseline']['accuracy']
    full_acc = ablation_summary['E1_Full']['accuracy']
    
    component_importance = {
        'Full Model Improvement': full_acc - baseline_acc,
        'Invariance Only': ablation_summary['E2_InvDyn']['accuracy'] - baseline_acc,
        'Covariance Only': ablation_summary['E7_CovDyn']['accuracy'] - baseline_acc,
        'Variance Only': ablation_summary['E8_VarDyn']['accuracy'] - baseline_acc,
        'Dynamic Weight Effect': ablation_summary['E1_Full']['accuracy'] - ablation_summary['E5_FullNoD']['accuracy']
    }
    
    with open(os.path.join(output_paths['ablation'], 'component_importance.json'), 'w') as f:
        json.dump(component_importance, f, indent=2)
    
    # Plot component importance
    plt.figure(figsize=(10, 6))
    components = list(component_importance.keys())
    importances = list(component_importance.values())
    colors = ['green' if x > 0 else 'red' for x in importances]
    
    plt.barh(components, importances, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Accuracy Improvement', fontsize=12)
    plt.title('Component Importance Analysis', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_paths['ablation'], 'component_importance.png')
    safe_plot_save(output_path, show=config.show_plots)
    
    for comp, imp in component_importance.items():
        logger.info(f"  {comp}: {imp:+.4f}")
    
    logger.info("\nAblation study completed")
    
    return {
        'ablation_results': ablation_results,
        'predictions': all_predictions,
        'model_names': all_model_names,
        'true_labels': true_labels,
        'component_importance': component_importance
    }


def run_mcnemar_test(config: ExperimentConfig, output_paths: Dict[str, str], 
                    ablation_data: Optional[Dict] = None):
    """Run McNemar's statistical significance testing"""
    logger.info("\n" + "="*80)
    logger.info("PHASE: McNEMAR'S SIGNIFICANCE TESTING")
    logger.info("="*80)
    
    # Check if McNemar's test is available
    if not ABLATION_STUDY_AVAILABLE:
        logger.error("McNemar's test module not available. Skipping significance testing.")
        return
    
    # If we don't have ablation data, we need to run ablation study first
    if ablation_data is None:
        logger.info("Running ablation study to get predictions...")
        ablation_data = run_ablation_study(config, output_paths)
    
    all_predictions = ablation_data['predictions']
    model_names = ablation_data['model_names']
    true_labels = ablation_data['true_labels']
    
    # Run pairwise McNemar's tests
    logger.info("\n>>> Running pairwise McNemar's tests")
    
    mcnemar_results = mcnemar_test_multiple(all_predictions, model_names, true_labels)
    
    # Save results
    with open(os.path.join(output_paths['mcnemar'], 'significance_tests.json'), 'w') as f:
        json.dump(mcnemar_results, f, indent=2)
    
    # Create pairwise comparison matrix
    logger.info("\n>>> Generating pairwise comparison matrix")
    
    n_models = len(model_names)
    p_value_matrix = np.ones((n_models, n_models))
    
    for comparison in mcnemar_results['pairwise_comparisons']:
        idx_a = model_names.index(comparison['algorithm_a'])
        idx_b = model_names.index(comparison['algorithm_b'])
        p_value_matrix[idx_a, idx_b] = comparison['p_value']
        p_value_matrix[idx_b, idx_a] = comparison['p_value']
    
    # Plot p-value matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(p_value_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r',
               xticklabels=model_names, yticklabels=model_names,
               vmin=0, vmax=0.1, center=0.05,
               cbar_kws={'label': 'p-value'})
    plt.title("McNemar's Test: Pairwise p-values\n(Green = significantly different, Red = not significant)")
    plt.tight_layout()
    
    output_path = os.path.join(output_paths['mcnemar'], 'pairwise_comparison_matrix.png')
    safe_plot_save(output_path, show=config.show_plots)
    
    # Generate contingency tables visualization
    logger.info("\n>>> Generating contingency tables")
    
    # Select key comparisons to visualize
    key_comparisons = [
        ('E6_Baseline', 'E1_Full'),  # Baseline vs Full model
        ('E1_Full', 'E5_FullNoD'),    # Full vs Full without dynamic
        ('E2_InvDyn', 'E7_CovDyn'),   # Invariance vs Covariance
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (model_a, model_b) in enumerate(key_comparisons):
        if model_a in model_names and model_b in model_names:
            # Find the comparison result
            comparison = next((c for c in mcnemar_results['pairwise_comparisons'] 
                             if (c['algorithm_a'] == model_a and c['algorithm_b'] == model_b) or
                                (c['algorithm_a'] == model_b and c['algorithm_b'] == model_a)), None)
            
            if comparison:
                n00, n01, n10, n11 = comparison['contingency_table']
                table = np.array([[n11, n10], [n01, n00]])
                
                ax = axes[idx]
                sns.heatmap(table, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=[f'{model_a}\nWrong', f'{model_a}\nCorrect'],
                           yticklabels=[f'{model_b}\nCorrect', f'{model_b}\nWrong'],
                           cbar=False)
                ax.set_title(f'{model_a} vs {model_b}\np={comparison["p_value"]:.4f}')
    
    plt.tight_layout()
    output_path = os.path.join(output_paths['mcnemar'], 'contingency_tables.png')
    safe_plot_save(output_path, show=config.show_plots)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("McNEMAR'S TEST SUMMARY")
    logger.info("="*80)
    
    significant_comparisons = [c for c in mcnemar_results['pairwise_comparisons'] 
                              if c['significant_at_0.05']]
    
    logger.info(f"\nTotal comparisons: {len(mcnemar_results['pairwise_comparisons'])}")
    logger.info(f"Significant differences (p < 0.05): {len(significant_comparisons)}")
    
    if significant_comparisons:
        logger.info("\nSignificant comparisons:")
        for comp in significant_comparisons[:10]:  # Show first 10
            logger.info(f"  {comp['algorithm_a']} vs {comp['algorithm_b']}: "
                       f"p={comp['p_value']:.4f} - {comp['effect_description']}")
    
    logger.info("\nMcNemar's testing completed")


def run_feature_analysis(config: ExperimentConfig, output_paths: Dict[str, str]):
    """Run comprehensive feature analysis including collapse detection"""
    logger.info("\n" + "="*80)
    logger.info("PHASE: FEATURE COLLAPSE ANALYSIS")
    logger.info("="*80)
    
    # Check if feature analysis is available
    if not FEATURE_ANALYSIS_AVAILABLE:
        logger.error("Feature analysis module not available. Skipping feature analysis.")
        return
    
    # Get data loaders
    _, _, test_loader = get_data_loaders(config)
    
    # Test baseline and proposed models with feature extraction
    logger.info("\n>>> Extracting features from baseline model")
    baseline_model = create_model('baseline', config)
    baseline_results = test_model(baseline_model, test_loader, config, extract_features=True)
    
    logger.info("\n>>> Extracting features from proposed model")
    proposed_model = create_model('proposed', config)
    proposed_results = test_model(proposed_model, test_loader, config, extract_features=True)
    
    feature_analysis_results = {}
    
    for model_name, results in [('baseline', baseline_results), ('proposed', proposed_results)]:
        if 'features' not in results or results['features'] is None:
            logger.warning(f"No features available for {model_name}")
            continue
        
        logger.info(f"\n>>> Analyzing {model_name} features")
        
        features = np.array(results['features'])
        labels = np.array(results['true_labels'])[:len(features)]
        
        # Feature collapse detection
        collapse_metrics = detect_feature_collapse(features)
        logger.info(f"  Collapsed dimensions: {collapse_metrics['collapsed_dimensions']}/{collapse_metrics['total_dimensions']}")
        logger.info(f"  Collapse ratio: {collapse_metrics['collapse_ratio']:.4f}")
        
        # Feature utilization
        utilization_metrics = compute_feature_utilization(features)
        logger.info(f"  Mean utilization: {utilization_metrics['mean_utilization']:.4f}")
        
        # Comprehensive analysis
        comprehensive_metrics = comprehensive_feature_analysis(features, labels)
        
        feature_analysis_results[model_name] = {
            'collapse': collapse_metrics,
            'utilization': utilization_metrics,
            'comprehensive': comprehensive_metrics
        }
        
        # Visualize feature variance distribution
        std_per_dim = collapse_metrics['std_per_dimension']
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(std_per_dim, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Number of Dimensions')
        plt.title(f'{model_name.capitalize()}: Feature Variance Distribution')
        plt.axvline(x=1e-4, color='r', linestyle='--', label='Collapse threshold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(sorted(std_per_dim, reverse=True))
        plt.xlabel('Dimension (sorted)')
        plt.ylabel('Standard Deviation')
        plt.title(f'{model_name.capitalize()}: Sorted Feature Variance')
        plt.yscale('log')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_paths['feature_analysis'], f'variance_distribution_{model_name}.png')
        safe_plot_save(output_path, show=config.show_plots)
        
        # Visualize correlation matrix (sample if too large)
        if features.shape[1] <= 100:
            corr_matrix = np.corrcoef(features.T)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                       vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})
            plt.title(f'{model_name.capitalize()}: Feature Correlation Matrix')
            plt.tight_layout()
            
            output_path = os.path.join(output_paths['feature_analysis'], f'correlation_matrix_{model_name}.png')
            safe_plot_save(output_path, show=config.show_plots)
        else:
            logger.info(f"  Skipping correlation matrix (too many dimensions: {features.shape[1]})")
    
    # Save feature analysis metrics (create a copy for JSON serialization)
    import copy
    save_results = copy.deepcopy(feature_analysis_results)
    
    # Remove std_per_dimension for JSON serialization (too large)
    for model_name in save_results:
        if 'std_per_dimension' in save_results[model_name]['collapse']:
            del save_results[model_name]['collapse']['std_per_dimension']
    
    with open(os.path.join(output_paths['feature_analysis'], 'feature_collapse_metrics.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logger.info("\nFeature analysis completed")


def main():
    """Main entry point for unified experiment runner"""
    parser = argparse.ArgumentParser(
        description='Unified Experiment Runner for Few-Shot Cosine Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dataset and model settings
    parser.add_argument('--dataset', type=str, default='miniImagenet',
                       help='Dataset (miniImagenet, CUB, CIFAR, Omniglot, etc.)')
    parser.add_argument('--backbone', type=str, default='Conv4',
                       help='Backbone architecture (Conv4, Conv6, ResNet10, ResNet18, ResNet34)')
    parser.add_argument('--n_way', type=int, default=5,
                       help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=1,
                       help='Number of support samples per class')
    parser.add_argument('--n_query', type=int, default=16,
                       help='Number of query samples per class')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--optimization', type=str, default='AdamW',
                       help='Optimizer (Adam, AdamW, SGD)')
    
    # Testing settings
    parser.add_argument('--test_iter', type=int, default=600,
                       help='Number of test episodes')
    
    # Run mode
    parser.add_argument('--run_mode', type=str, default='all',
                       choices=['all', 'train_test', 'ablation', 'qualitative', 'feature_analysis', 'mcnemar'],
                       help='Select which experiments to run')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/Few-Shot-Cosine-Transformer/record',
                       help='Directory for saving results and visualizations')
    parser.add_argument('--seed', type=int, default=4040,
                       help='Random seed for reproducibility')
    
    # Checkpoint settings
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                       help='Path to baseline model checkpoint (skip training if provided)')
    parser.add_argument('--proposed_checkpoint', type=str, default=None,
                       help='Path to proposed model checkpoint (skip training if provided)')
    
    # Ablation settings
    parser.add_argument('--ablation_experiments', type=str, default=None,
                       help='Comma-separated list of ablation experiments to run (e.g., E1,E2,E3)')
    
    # Visualization settings
    parser.add_argument('--show_plots', action='store_true', default=True,
                       help='Display plots interactively during execution (default: True)')
    
    # McNemar testing settings
    parser.add_argument('--mcnemar_each_test', action='store_true', default=True,
                       help='Run McNemar statistical significance test after each testing phase (default: True)')
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = ExperimentConfig(
        dataset=args.dataset,
        backbone=args.backbone,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimization=args.optimization,
        test_iter=args.test_iter,
        run_mode=RunMode(args.run_mode),
        output_dir=args.output_dir,
        seed=args.seed,
        baseline_checkpoint=args.baseline_checkpoint,
        proposed_checkpoint=args.proposed_checkpoint,
        show_plots=args.show_plots,
        mcnemar_each_test=args.mcnemar_each_test
    )
    
    # Parse ablation experiments if specified
    if args.ablation_experiments:
        config.ablation_experiments = [e.strip() for e in args.ablation_experiments.split(',')]
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup output directories
    output_paths = setup_output_directories(config)
    
    # Save configuration
    with open(os.path.join(output_paths['base'], 'experiment_config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Log experiment info
    logger.info("\n" + "="*80)
    logger.info("UNIFIED EXPERIMENT RUNNER")
    logger.info("="*80)
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Backbone: {config.backbone}")
    logger.info(f"Few-shot setting: {config.n_way}-way {config.k_shot}-shot")
    logger.info(f"Run mode: {config.run_mode.value}")
    logger.info(f"Output directory: {output_paths['base']}")
    logger.info(f"Device: {device}")
    
    # Run experiments based on mode
    start_time = time.time()
    
    train_test_results = None
    ablation_data = None
    
    if config.run_mode == RunMode.ALL or config.run_mode == RunMode.TRAIN_TEST:
        train_test_results = run_train_test(config, output_paths)
    
    if config.run_mode == RunMode.ALL or config.run_mode == RunMode.QUALITATIVE:
        run_qualitative_analysis(config, output_paths, train_test_results)
    
    if config.run_mode == RunMode.ALL or config.run_mode == RunMode.ABLATION:
        ablation_data = run_ablation_study(config, output_paths)
    
    if config.run_mode == RunMode.ALL or config.run_mode == RunMode.MCNEMAR:
        run_mcnemar_test(config, output_paths, ablation_data)
    
    if config.run_mode == RunMode.ALL or config.run_mode == RunMode.FEATURE_ANALYSIS:
        run_feature_analysis(config, output_paths)
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("="*80)
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    logger.info(f"Results saved to: {output_paths['base']}")
    logger.info("\nThank you for using the Unified Experiment Runner!")


if __name__ == '__main__':
    main()
