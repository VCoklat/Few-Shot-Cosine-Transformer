#!/bin/bash
# Example training scripts with VIC regularization
# This script demonstrates various configurations for using VIC regularization

echo "=============================================="
echo "VIC Regularization Training Examples"
echo "=============================================="
echo ""

# Example 1: Basic VIC training with miniImagenet
echo "Example 1: Basic VIC training"
echo "Command: python train.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 5 --use_vic 1 --num_epoch 10"
echo ""

# Example 2: VIC with memory optimization (mixed precision)
echo "Example 2: VIC with mixed precision (FP16) for memory efficiency"
echo "Command: python train.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 5 --use_vic 1 --mixed_precision 1 --num_epoch 10"
echo ""

# Example 3: VIC with custom weights
echo "Example 3: VIC with custom initial weights"
echo "Command: python train.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 5 --use_vic 1 --vic_lambda_v 0.5 --vic_lambda_i 1.0 --vic_lambda_c 1.5 --num_epoch 10"
echo ""

# Example 4: VIC with smaller backbone (Conv4) for faster training
echo "Example 4: VIC with Conv4 backbone (faster, less memory)"
echo "Command: python train.py --method FSCT_cosine --dataset miniImagenet --backbone Conv4 --n_way 5 --k_shot 5 --use_vic 1 --num_epoch 10"
echo ""

# Example 5: VIC with data augmentation
echo "Example 5: VIC with data augmentation"
echo "Command: python train.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 5 --use_vic 1 --train_aug 1 --num_epoch 10"
echo ""

# Example 6: VIC with WandB logging
echo "Example 6: VIC with WandB logging for monitoring"
echo "Command: python train.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 5 --use_vic 1 --wandb 1 --num_epoch 10"
echo ""

# Example 7: Full configuration for Kaggle (16GB VRAM)
echo "Example 7: Optimized for Kaggle 16GB VRAM"
echo "Command: python train.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 5 --use_vic 1 --mixed_precision 1 --n_episode 100 --n_query 8 --num_epoch 50 --wandb 1"
echo ""

# Example 8: 1-shot learning with VIC
echo "Example 8: 1-shot learning (more challenging)"
echo "Command: python train.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 1 --use_vic 1 --mixed_precision 1 --num_epoch 10"
echo ""

echo "=============================================="
echo "To run any example, copy the command and execute it"
echo "Note: Adjust num_epoch, dataset paths in configs.py as needed"
echo "=============================================="
echo ""
echo "For more information, see VIC_REGULARIZATION.md"
