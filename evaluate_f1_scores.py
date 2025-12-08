"""
F1 Score Evaluation for Enhanced Few-Shot Learning

This script evaluates the model using F1 score metrics per class and overall,
providing detailed classification reports.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import argparse
import tqdm
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backbone
import configs
from data.datamgr import SetDataManager
from io_utils import model_dict
from models.optimal_fewshot_enhanced import EnhancedOptimalFewShot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_f1_scores(model, data_loader, n_way):
    """
    Evaluate model and compute F1 scores per class.
    
    Args:
        model: Trained model
        data_loader: DataLoader for test episodes
        n_way: Number of classes
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    episode_results = []
    
    with torch.no_grad():
        for episode_idx, (x, _) in enumerate(tqdm.tqdm(data_loader, desc='Evaluating')):
            x = x.to(device)
            
            # Forward pass
            logits = model.set_forward(x)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Get targets
            targets = np.repeat(range(n_way), logits.size(0) // n_way)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Episode accuracy
            episode_acc = (predictions == targets).mean()
            episode_results.append(episode_acc)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    results = {
        'accuracy': np.mean(episode_results),
        'accuracy_std': np.std(episode_results),
        'accuracy_95ci': 1.96 * np.std(episode_results) / np.sqrt(len(episode_results)),
        
        # Per-class metrics
        'f1_per_class': {},
        'precision_per_class': {},
        'recall_per_class': {},
        
        # Overall metrics
        'f1_macro': f1_score(all_targets, all_predictions, average='macro'),
        'f1_micro': f1_score(all_targets, all_predictions, average='micro'),
        'f1_weighted': f1_score(all_targets, all_predictions, average='weighted'),
        
        'precision_macro': precision_score(all_targets, all_predictions, average='macro'),
        'recall_macro': recall_score(all_targets, all_predictions, average='macro'),
        
        # Per-class F1 scores
        'f1_scores': f1_score(all_targets, all_predictions, average=None),
        'precision_scores': precision_score(all_targets, all_predictions, average=None),
        'recall_scores': recall_score(all_targets, all_predictions, average=None),
        
        # Confusion matrix
        'confusion_matrix': confusion_matrix(all_targets, all_predictions),
        
        # Classification report
        'classification_report': classification_report(
            all_targets, all_predictions, 
            target_names=[f'Class {i}' for i in range(n_way)],
            output_dict=True
        )
    }
    
    # Organize per-class results
    for i in range(n_way):
        results['f1_per_class'][f'Class_{i}'] = results['f1_scores'][i]
        results['precision_per_class'][f'Class_{i}'] = results['precision_scores'][i]
        results['recall_per_class'][f'Class_{i}'] = results['recall_scores'][i]
    
    return results


def print_results(results, n_way):
    """
    Print formatted evaluation results.
    
    Args:
        results: Dictionary with metrics
        n_way: Number of classes
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Overall accuracy
    print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}% ± {results['accuracy_95ci']*100:.2f}%")
    print(f"  (Mean ± 95% CI over episodes)")
    
    # Overall metrics
    print("\n" + "-" * 80)
    print("Overall Metrics:")
    print("-" * 80)
    print(f"  F1 Score (Macro):    {results['f1_macro']:.4f}")
    print(f"  F1 Score (Micro):    {results['f1_micro']:.4f}")
    print(f"  F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"  Precision (Macro):   {results['precision_macro']:.4f}")
    print(f"  Recall (Macro):      {results['recall_macro']:.4f}")
    
    # Per-class metrics
    print("\n" + "-" * 80)
    print("Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<10} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    for i in range(n_way):
        f1 = results['f1_scores'][i]
        precision = results['precision_scores'][i]
        recall = results['recall_scores'][i]
        print(f"Class {i:<4} {f1:.4f}       {precision:.4f}       {recall:.4f}")
    
    # Average
    print("-" * 80)
    print(f"{'Average':<10} {results['f1_macro']:.4f}       {results['precision_macro']:.4f}       {results['recall_macro']:.4f}")
    
    # Confusion matrix
    print("\n" + "-" * 80)
    print("Confusion Matrix:")
    print("-" * 80)
    cm = results['confusion_matrix']
    
    # Header
    print(f"{'Pred/True':<12}", end='')
    for i in range(n_way):
        print(f"Class {i:<4}", end=' ')
    print()
    print("-" * 80)
    
    # Rows
    for i in range(n_way):
        print(f"Class {i:<6}", end=' ')
        for j in range(n_way):
            print(f"{cm[i, j]:<10}", end=' ')
        print()
    
    print("=" * 80)


def save_results(results, output_path):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary with metrics
        output_path: Path to save results
    """
    # Convert numpy arrays to lists for JSON serialization
    results_json = {
        'accuracy': float(results['accuracy']),
        'accuracy_std': float(results['accuracy_std']),
        'accuracy_95ci': float(results['accuracy_95ci']),
        
        'f1_macro': float(results['f1_macro']),
        'f1_micro': float(results['f1_micro']),
        'f1_weighted': float(results['f1_weighted']),
        'precision_macro': float(results['precision_macro']),
        'recall_macro': float(results['recall_macro']),
        
        'f1_per_class': {k: float(v) for k, v in results['f1_per_class'].items()},
        'precision_per_class': {k: float(v) for k, v in results['precision_per_class'].items()},
        'recall_per_class': {k: float(v) for k, v in results['recall_per_class'].items()},
        
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report']
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function for F1 score evaluation"""
    parser = argparse.ArgumentParser(description='F1 Score Evaluation for Enhanced Few-Shot Learning')
    
    # Dataset and model
    parser.add_argument('--dataset', default='miniImagenet', 
                       help='Dataset: Omniglot/CUB/miniImagenet/HAM10000')
    parser.add_argument('--backbone', default='Conv4',
                       help='Backbone: Conv4/ResNet18/ResNet34')
    parser.add_argument('--n_way', default=5, type=int,
                       help='Number of classes per episode')
    parser.add_argument('--k_shot', default=1, type=int,
                       help='Number of support samples per class')
    parser.add_argument('--n_query', default=16, type=int,
                       help='Number of query samples per class')
    
    # Model architecture
    parser.add_argument('--feature_dim', default=64, type=int,
                       help='Feature dimension for transformer')
    parser.add_argument('--n_heads', default=4, type=int,
                       help='Number of attention heads')
    parser.add_argument('--dropout', default=0.1, type=float,
                       help='Dropout rate')
    
    # Evaluation parameters
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--n_episodes', default=600, type=int,
                       help='Number of test episodes')
    parser.add_argument('--output_dir', default='./evaluation_results',
                       help='Output directory for results')
    
    # Configuration parameters
    parser.add_argument('--use_task_invariance', default=1, type=int,
                       help='Use task-adaptive invariance')
    parser.add_argument('--use_multi_scale', default=1, type=int,
                       help='Use multi-scale invariance')
    parser.add_argument('--use_feature_augmentation', default=1, type=int,
                       help='Use feature augmentation')
    parser.add_argument('--use_prototype_refinement', default=0, type=int,
                       help='Use prototype refinement')
    parser.add_argument('--domain', default='general',
                       help='Domain: general/medical/fine_grained')
    parser.add_argument('--split', default='novel',
                       help='Data split: base/val/novel')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("F1 Score Evaluation for Enhanced Few-Shot Learning")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Task: {args.n_way}-way {args.k_shot}-shot")
    print(f"Test episodes: {args.n_episodes}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)
    
    # Load data
    print("\nLoading test data...")
    
    # Determine test file
    split = args.split
    if args.dataset == 'cross':
        if split == 'base':
            testfile = configs.data_dir['miniImagenet'] + 'all.json'
        else:
            testfile = configs.data_dir['CUB'] + split + '.json'
    elif args.dataset == 'cross_char':
        if split == 'base':
            testfile = configs.data_dir['Omniglot'] + 'noLatin.json'
        else:
            testfile = configs.data_dir['emnist'] + split + '.json'
    else:
        testfile = configs.data_dir[args.dataset] + split + '.json'
    
    # Determine image size
    if args.dataset == "CIFAR":
        image_size = 112 if 'ResNet' in args.backbone else 64
    else:
        image_size = 224 if 'ResNet' in args.backbone else 84
    
    datamgr = SetDataManager(
        image_size,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_episode=args.n_episodes
    )
    data_loader = datamgr.get_data_loader(testfile, aug=False)
    
    # Create model
    print("\nCreating model...")
    if args.backbone in model_dict:
        model_func = model_dict[args.backbone]
    else:
        print(f"Warning: Unknown backbone {args.backbone}, using Conv4")
        model_func = backbone.Conv4
    
    model = EnhancedOptimalFewShot(
        model_func=model_func,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        feature_dim=args.feature_dim,
        n_heads=args.n_heads,
        dropout=args.dropout,
        dataset=args.dataset,
        use_task_invariance=bool(args.use_task_invariance),
        use_multi_scale=bool(args.use_multi_scale),
        use_feature_augmentation=bool(args.use_feature_augmentation),
        use_prototype_refinement=bool(args.use_prototype_refinement),
        domain=args.domain
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state' in checkpoint:
        model.load_state_dict(checkpoint['state'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_f1_scores(model, data_loader, args.n_way)
    
    # Print results
    print_results(results, args.n_way)
    
    # Save results
    output_path = os.path.join(
        args.output_dir, 
        f'f1_scores_{args.dataset}_{args.n_way}way_{args.k_shot}shot.json'
    )
    save_results(results, output_path)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
