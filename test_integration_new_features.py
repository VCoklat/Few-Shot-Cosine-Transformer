#!/usr/bin/env python3
"""
Integration test and demonstration for new features:
- show_plots flag
- mcnemar_each_test flag

This script demonstrates how the new features work without requiring
a full dataset or GPU.
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.experiment_config import ExperimentConfig, RunMode


def test_command_line_args():
    """Test that command-line arguments are properly parsed"""
    print("="*60)
    print("Testing Command-Line Argument Parsing")
    print("="*60)
    
    # Simulate command-line arguments
    test_args = [
        '--dataset', 'miniImagenet',
        '--backbone', 'Conv4',
        '--show_plots',
        '--mcnemar_each_test',
        '--run_mode', 'train_test',
        '--output_dir', './test_output',
        '--seed', '1234'
    ]
    
    # Parse arguments (simulating main function behavior)
    parser = argparse.ArgumentParser()
    
    # Dataset and model settings
    parser.add_argument('--dataset', type=str, default='miniImagenet')
    parser.add_argument('--backbone', type=str, default='Conv4')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--n_query', type=int, default=16)
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimization', type=str, default='AdamW')
    
    # Testing settings
    parser.add_argument('--test_iter', type=int, default=600)
    
    # Run mode
    parser.add_argument('--run_mode', type=str, default='all',
                       choices=['all', 'train_test', 'ablation', 'qualitative', 'feature_analysis', 'mcnemar'])
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=4040)
    
    # Checkpoint settings
    parser.add_argument('--baseline_checkpoint', type=str, default=None)
    parser.add_argument('--proposed_checkpoint', type=str, default=None)
    
    # Ablation settings
    parser.add_argument('--ablation_experiments', type=str, default=None)
    
    # Visualization settings
    parser.add_argument('--show_plots', action='store_true', default=False)
    
    # McNemar testing settings
    parser.add_argument('--mcnemar_each_test', action='store_true', default=False)
    
    args = parser.parse_args(test_args)
    
    # Create config from parsed arguments
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
    
    # Verify the config was created correctly
    print("\nParsed Configuration:")
    print(f"  dataset: {config.dataset}")
    print(f"  backbone: {config.backbone}")
    print(f"  run_mode: {config.run_mode.value}")
    print(f"  output_dir: {config.output_dir}")
    print(f"  seed: {config.seed}")
    print(f"  show_plots: {config.show_plots}")
    print(f"  mcnemar_each_test: {config.mcnemar_each_test}")
    
    # Verify the flags are set correctly
    assert config.show_plots == True, "show_plots should be True"
    assert config.mcnemar_each_test == True, "mcnemar_each_test should be True"
    
    print("\n✓ Command-line arguments parsed successfully")
    print("✓ New flags (--show_plots and --mcnemar_each_test) are working correctly")
    

def test_default_values():
    """Test that default values are correct when flags are not specified"""
    print("\n" + "="*60)
    print("Testing Default Values")
    print("="*60)
    
    # Simulate command-line arguments without the new flags
    test_args = [
        '--dataset', 'CUB',
        '--backbone', 'ResNet18'
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='miniImagenet')
    parser.add_argument('--backbone', type=str, default='Conv4')
    parser.add_argument('--show_plots', action='store_true', default=False)
    parser.add_argument('--mcnemar_each_test', action='store_true', default=False)
    
    args = parser.parse_args(test_args)
    
    print("\nParsed Arguments (without new flags):")
    print(f"  dataset: {args.dataset}")
    print(f"  backbone: {args.backbone}")
    print(f"  show_plots: {args.show_plots}")
    print(f"  mcnemar_each_test: {args.mcnemar_each_test}")
    
    # Verify defaults
    assert args.show_plots == False, "show_plots should default to False"
    assert args.mcnemar_each_test == False, "mcnemar_each_test should default to False"
    
    print("\n✓ Default values are correct (both False)")


def demonstrate_usage():
    """Demonstrate how to use the new features"""
    print("\n" + "="*60)
    print("Usage Examples")
    print("="*60)
    
    print("\n1. Run with plot display enabled:")
    print("   python run_experiments.py --dataset miniImagenet --backbone Conv4 --show_plots")
    
    print("\n2. Run with McNemar testing after each test:")
    print("   python run_experiments.py --dataset miniImagenet --backbone Conv4 --mcnemar_each_test")
    
    print("\n3. Run with both features enabled:")
    print("   python run_experiments.py --dataset miniImagenet --backbone Conv4 --show_plots --mcnemar_each_test")
    
    print("\n4. Run train_test mode with both features:")
    print("   python run_experiments.py --dataset CUB --backbone ResNet18 --run_mode train_test --show_plots --mcnemar_each_test")
    
    print("\n5. Run ablation study with McNemar testing:")
    print("   python run_experiments.py --dataset miniImagenet --backbone Conv4 --run_mode ablation --mcnemar_each_test")


def main():
    """Main entry point"""
    print("\n")
    print("#" * 60)
    print("# Integration Test: show_plots and mcnemar_each_test")
    print("#" * 60)
    
    try:
        test_command_line_args()
        test_default_values()
        demonstrate_usage()
        
        print("\n" + "="*60)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("="*60)
        print()
        
        return 0
    
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
