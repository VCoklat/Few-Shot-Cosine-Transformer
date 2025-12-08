"""
Compare Two Model Branches Using McNemar's Test

This script compares two model configurations (branches) using McNemar's statistical test
to determine if there is a statistically significant difference in their performance.

Usage:
    python compare_models_mcnemar.py \
        --dataset miniImagenet \
        --method_a FSCT_cosine \
        --method_b FSCT_softmax \
        --checkpoint_a ./checkpoints/model_a.tar \
        --checkpoint_b ./checkpoints/model_b.tar \
        --n_way 5 \
        --k_shot 1 \
        --n_query 16 \
        --test_iter 600

Author: Few-Shot Learning Team
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backbone
import configs
from data.datamgr import SetDataManager
from io_utils import model_dict, get_best_file, get_assigned_file
from methods.CTX import CTX
from methods.transformer import FewShotTransformer
from eval_utils import evaluate_with_predictions, compare_models_mcnemar, print_mcnemar_comparison

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def change_model(model_name):
    """Adjust model name for specific datasets"""
    if model_name == 'Conv4':
        model_name = 'Conv4NP'
    elif model_name == 'Conv6':
        model_name = 'Conv6NP'
    elif model_name == 'Conv4S':
        model_name = 'Conv4SNP'
    elif model_name == 'Conv6S':
        model_name = 'Conv6SNP'
    return model_name


def load_model(method, backbone, dataset, n_way, k_shot, n_query, checkpoint_path=None, 
               checkpoint_dir=None, feti=False):
    """
    Load a model with specified configuration and checkpoint.
    
    Args:
        method: Method name (e.g., 'FSCT_cosine', 'CTX_softmax')
        backbone: Backbone architecture name
        dataset: Dataset name
        n_way: Number of ways
        k_shot: Number of shots
        n_query: Number of query samples
        checkpoint_path: Direct path to checkpoint file
        checkpoint_dir: Directory containing checkpoints (will use best_model.tar)
        feti: Use FETI pretrained backbone
    
    Returns:
        Loaded model
    """
    few_shot_params = dict(n_way=n_way, k_shot=k_shot, n_query=n_query)
    
    # Create model based on method
    if method in ['FSCT_softmax', 'FSCT_cosine']:
        variant = 'cosine' if method == 'FSCT_cosine' else 'softmax'
        
        def feature_model():
            if dataset in ['Omniglot', 'cross_char']:
                backbone_name = change_model(backbone)
            else:
                backbone_name = backbone
            
            if 'ResNet' in backbone_name:
                return model_dict[backbone_name](feti, dataset, flatten=True)
            else:
                return model_dict[backbone_name](dataset, flatten=True)
        
        model = FewShotTransformer(feature_model, variant=variant, **few_shot_params)
        
    elif method in ['CTX_softmax', 'CTX_cosine']:
        variant = 'cosine' if method == 'CTX_cosine' else 'softmax'
        input_dim = 512 if "ResNet" in backbone else 64
        
        def feature_model():
            if dataset in ['Omniglot', 'cross_char']:
                backbone_name = change_model(backbone)
            else:
                backbone_name = backbone
            
            if 'ResNet' in backbone_name:
                return model_dict[backbone_name](feti, dataset, flatten=False)
            else:
                return model_dict[backbone_name](dataset, flatten=False)
        
        model = CTX(feature_model, variant=variant, input_dim=input_dim, **few_shot_params)
    else:
        raise ValueError(f'Unknown method: {method}')
    
    model = model.to(device)
    
    # Load checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    elif checkpoint_dir and os.path.isdir(checkpoint_dir):
        modelfile = get_best_file(checkpoint_dir)
        if modelfile is None:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")
        print(f"Loading best checkpoint from: {modelfile}")
        checkpoint = torch.load(modelfile, map_location=device, weights_only=False)
    else:
        raise ValueError("Must provide either checkpoint_path or checkpoint_dir")
    
    # Load state dict
    if 'state' in checkpoint:
        model.load_state_dict(checkpoint['state'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Compare two model branches using McNemar\'s test'
    )
    
    # Dataset and task configuration
    parser.add_argument('--dataset', default='miniImagenet',
                       help='Dataset: CIFAR/CUB/miniImagenet/Omniglot/HAM10000/etc.')
    parser.add_argument('--backbone', default='Conv4',
                       help='Backbone: Conv4/Conv6/ResNet12/ResNet18/ResNet34')
    parser.add_argument('--n_way', default=5, type=int,
                       help='Number of classes per episode')
    parser.add_argument('--k_shot', default=1, type=int,
                       help='Number of support samples per class')
    parser.add_argument('--n_query', default=16, type=int,
                       help='Number of query samples per class')
    parser.add_argument('--test_iter', default=600, type=int,
                       help='Number of test episodes')
    
    # Model A configuration
    parser.add_argument('--method_a', default='FSCT_cosine',
                       help='Method for model A (e.g., FSCT_cosine, CTX_softmax)')
    parser.add_argument('--checkpoint_a', type=str, default=None,
                       help='Path to checkpoint file for model A')
    parser.add_argument('--checkpoint_dir_a', type=str, default=None,
                       help='Directory containing checkpoints for model A (uses best_model.tar)')
    parser.add_argument('--name_a', type=str, default=None,
                       help='Display name for model A (defaults to method_a)')
    
    # Model B configuration
    parser.add_argument('--method_b', default='FSCT_softmax',
                       help='Method for model B (e.g., FSCT_cosine, CTX_softmax)')
    parser.add_argument('--checkpoint_b', type=str, default=None,
                       help='Path to checkpoint file for model B')
    parser.add_argument('--checkpoint_dir_b', type=str, default=None,
                       help='Directory containing checkpoints for model B (uses best_model.tar)')
    parser.add_argument('--name_b', type=str, default=None,
                       help='Display name for model B (defaults to method_b)')
    
    # Other options
    parser.add_argument('--split', default='novel',
                       help='Dataset split: base/val/novel')
    parser.add_argument('--feti', type=int, default=0,
                       help='Use FETI pretrained backbone (for ResNet only)')
    parser.add_argument('--output', type=str, default='./record/mcnemar_comparison.json',
                       help='Output path for comparison results')
    
    args = parser.parse_args()
    
    # Validate that checkpoints are provided
    if not (args.checkpoint_a or args.checkpoint_dir_a):
        raise ValueError("Must provide either --checkpoint_a or --checkpoint_dir_a for model A")
    if not (args.checkpoint_b or args.checkpoint_dir_b):
        raise ValueError("Must provide either --checkpoint_b or --checkpoint_dir_b for model B")
    
    # Set display names
    model_a_name = args.name_a if args.name_a else args.method_a
    model_b_name = args.name_b if args.name_b else args.method_b
    
    print("=" * 80)
    print("McNEMAR'S TEST MODEL COMPARISON")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Task: {args.n_way}-way {args.k_shot}-shot")
    print(f"Test episodes: {args.test_iter}")
    print(f"\nModel A: {model_a_name} ({args.method_a})")
    if args.checkpoint_a:
        print(f"  Checkpoint: {args.checkpoint_a}")
    else:
        print(f"  Checkpoint dir: {args.checkpoint_dir_a}")
    print(f"\nModel B: {model_b_name} ({args.method_b})")
    if args.checkpoint_b:
        print(f"  Checkpoint: {args.checkpoint_b}")
    else:
        print(f"  Checkpoint dir: {args.checkpoint_dir_b}")
    print("=" * 80)
    
    # Determine image size
    if args.dataset == "CIFAR":
        image_size = 112 if 'ResNet' in args.backbone else 64
    else:
        image_size = 224 if 'ResNet' in args.backbone else 84
    
    # Get test file
    if args.dataset == 'cross':
        if args.split == 'base':
            testfile = configs.data_dir['miniImagenet'] + 'all.json'
        else:
            testfile = configs.data_dir['CUB'] + args.split + '.json'
    elif args.dataset == 'cross_char':
        if args.split == 'base':
            testfile = configs.data_dir['Omniglot'] + 'noLatin.json'
        else:
            testfile = configs.data_dir['emnist'] + args.split + '.json'
    else:
        testfile = configs.data_dir[args.dataset] + args.split + '.json'
    
    print(f"\nTest file: {testfile}")
    
    # Create data loader
    print("\nCreating data loader...")
    test_datamgr = SetDataManager(
        image_size, 
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_episode=args.test_iter
    )
    test_loader = test_datamgr.get_data_loader(testfile, aug=False)
    
    # Load Model A
    print(f"\n{'='*80}")
    print(f"Loading Model A: {model_a_name}")
    print(f"{'='*80}")
    model_a = load_model(
        args.method_a,
        args.backbone,
        args.dataset,
        args.n_way,
        args.k_shot,
        args.n_query,
        checkpoint_path=args.checkpoint_a,
        checkpoint_dir=args.checkpoint_dir_a,
        feti=bool(args.feti)
    )
    
    # Load Model B
    print(f"\n{'='*80}")
    print(f"Loading Model B: {model_b_name}")
    print(f"{'='*80}")
    model_b = load_model(
        args.method_b,
        args.backbone,
        args.dataset,
        args.n_way,
        args.k_shot,
        args.n_query,
        checkpoint_path=args.checkpoint_b,
        checkpoint_dir=args.checkpoint_dir_b,
        feti=bool(args.feti)
    )
    
    # Evaluate Model A
    print(f"\n{'='*80}")
    print(f"Evaluating Model A: {model_a_name}")
    print(f"{'='*80}")
    results_a, predictions_a, true_labels = evaluate_with_predictions(
        test_loader, model_a, args.n_way, device=device
    )
    print(f"Model A Accuracy: {results_a['accuracy']:.4f} ({results_a['accuracy']*100:.2f}%)")
    if 'confidence_interval_95' in results_a:
        ci = results_a['confidence_interval_95']
        print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}] (±{ci['margin']:.4f})")
    
    # Evaluate Model B
    print(f"\n{'='*80}")
    print(f"Evaluating Model B: {model_b_name}")
    print(f"{'='*80}")
    results_b, predictions_b, _ = evaluate_with_predictions(
        test_loader, model_b, args.n_way, device=device
    )
    print(f"Model B Accuracy: {results_b['accuracy']:.4f} ({results_b['accuracy']*100:.2f}%)")
    if 'confidence_interval_95' in results_b:
        ci = results_b['confidence_interval_95']
        print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}] (±{ci['margin']:.4f})")
    
    # Perform McNemar's test
    print(f"\n{'='*80}")
    print("Performing McNemar's Test")
    print(f"{'='*80}")
    mcnemar_result = compare_models_mcnemar(
        predictions_a,
        predictions_b,
        true_labels,
        model_a_name=model_a_name,
        model_b_name=model_b_name
    )
    
    # Print comparison results
    print_mcnemar_comparison(mcnemar_result)
    
    # Save results to JSON
    output_data = {
        'configuration': {
            'dataset': args.dataset,
            'backbone': args.backbone,
            'n_way': args.n_way,
            'k_shot': args.k_shot,
            'n_query': args.n_query,
            'test_episodes': args.test_iter,
            'split': args.split
        },
        'model_a': {
            'name': model_a_name,
            'method': args.method_a,
            'checkpoint': args.checkpoint_a or args.checkpoint_dir_a,
            'results': {
                'accuracy': float(results_a['accuracy']),
                'confidence_interval_95': results_a.get('confidence_interval_95', {}),
                'kappa': float(results_a.get('kappa', 0)),
                'mcc': float(results_a.get('mcc', 0))
            }
        },
        'model_b': {
            'name': model_b_name,
            'method': args.method_b,
            'checkpoint': args.checkpoint_b or args.checkpoint_dir_b,
            'results': {
                'accuracy': float(results_b['accuracy']),
                'confidence_interval_95': results_b.get('confidence_interval_95', {}),
                'kappa': float(results_b.get('kappa', 0)),
                'mcc': float(results_b.get('mcc', 0))
            }
        },
        'mcnemar_test': {
            'contingency_table': mcnemar_result['contingency_table'],
            'statistic': float(mcnemar_result['statistic']),
            'p_value': float(mcnemar_result['p_value']),
            'significant_at_0.05': bool(mcnemar_result['significant_at_0.05']),
            'significant_at_0.01': bool(mcnemar_result['significant_at_0.01']),
            'algorithm_a_better': bool(mcnemar_result['algorithm_a_better']),
            'algorithm_b_better': bool(mcnemar_result['algorithm_b_better']),
            'discordant_pairs': int(mcnemar_result['discordant_pairs']),
            'effect_description': mcnemar_result['effect_description'],
            'test_type': mcnemar_result['test_type']
        }
    }
    
    # Create output directory if it doesn't exist
    # os.path.dirname() returns empty string when path has no directory component
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only create directory if output path contains a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{model_a_name}: {results_a['accuracy']*100:.2f}%")
    print(f"{model_b_name}: {results_b['accuracy']*100:.2f}%")
    print(f"Accuracy difference: {abs(results_a['accuracy'] - results_b['accuracy'])*100:.2f}%")
    
    if mcnemar_result['significant_at_0.05']:
        if mcnemar_result['algorithm_a_better']:
            print(f"\n✓ {model_a_name} performs SIGNIFICANTLY BETTER than {model_b_name}")
        else:
            print(f"\n✓ {model_b_name} performs SIGNIFICANTLY BETTER than {model_a_name}")
        print(f"  (p-value: {mcnemar_result['p_value']:.6f} < 0.05)")
    else:
        print(f"\n○ No statistically significant difference detected")
        print(f"  (p-value: {mcnemar_result['p_value']:.6f} >= 0.05)")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
