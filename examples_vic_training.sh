#!/bin/bash
# Example training scripts for VIC-Enhanced FS-CT

echo "VIC-Enhanced Few-Shot Cosine Transformer Training Examples"
echo "============================================================"
echo ""

# Example 1: Basic 5-way 5-shot training with VIC loss on miniImagenet
echo "Example 1: Basic VIC training (5-way 5-shot, miniImagenet)"
echo "-----------------------------------------------------------"
echo "python train.py --method FSCT_cosine --dataset miniImagenet \\"
echo "  --backbone ResNet18 --n_way 5 --k_shot 5 \\"
echo "  --use_vic_loss 1 --lambda_v 1.0 --lambda_i 1.0 --lambda_c 0.04 \\"
echo "  --num_epoch 50 --learning_rate 1e-3 --n_episode 200"
echo ""

# Example 2: 5-way 1-shot with VIC loss on CUB
echo "Example 2: Few-shot with VIC (5-way 1-shot, CUB)"
echo "-----------------------------------------------------------"
echo "python train.py --method FSCT_cosine --dataset CUB \\"
echo "  --backbone ResNet34 --n_way 5 --k_shot 1 \\"
echo "  --use_vic_loss 1 --lambda_v 2.0 --lambda_i 1.0 --lambda_c 0.02 \\"
echo "  --num_epoch 50 --train_aug 1"
echo ""

# Example 3: Standard training without VIC loss (for comparison)
echo "Example 3: Standard training without VIC (for comparison)"
echo "-----------------------------------------------------------"
echo "python train.py --method FSCT_cosine --dataset miniImagenet \\"
echo "  --backbone ResNet18 --n_way 5 --k_shot 5 \\"
echo "  --use_vic_loss 0 --num_epoch 50"
echo ""

# Example 4: Training with different VIC weights
echo "Example 4: Custom VIC weights (emphasis on variance)"
echo "-----------------------------------------------------------"
echo "python train.py --method FSCT_cosine --dataset CIFAR \\"
echo "  --backbone Conv4 --n_way 5 --k_shot 5 \\"
echo "  --use_vic_loss 1 --lambda_v 2.5 --lambda_i 1.0 --lambda_c 0.03 \\"
echo "  --num_epoch 50"
echo ""

# Example 5: Full training and testing pipeline
echo "Example 5: Full train-test pipeline with VIC"
echo "-----------------------------------------------------------"
echo "python train_test.py --method FSCT_cosine --dataset miniImagenet \\"
echo "  --backbone ResNet18 --n_way 5 --k_shot 5 \\"
echo "  --use_vic_loss 1 --lambda_v 1.0 --lambda_i 1.0 --lambda_c 0.04 \\"
echo "  --num_epoch 50 --test_iter 600 --wandb 1"
echo ""

# Example 6: Memory-efficient training for 8GB GPU
echo "Example 6: Memory-efficient training (optimized for 8GB GPU)"
echo "-----------------------------------------------------------"
echo "python train.py --method FSCT_cosine --dataset miniImagenet \\"
echo "  --backbone Conv6 --n_way 5 --k_shot 5 \\"
echo "  --use_vic_loss 1 --lambda_v 1.0 --lambda_i 1.0 --lambda_c 0.04 \\"
echo "  --num_epoch 50 --n_episode 150 --n_query 12"
echo ""

echo "============================================================"
echo "To run any example, copy the command (without the backslashes)"
echo "or uncomment the corresponding line below and execute this script."
echo ""
echo "For more details, see VIC_LOSS_README.md"
echo "============================================================"

# Uncomment to run a specific example:

# Example 1 - Recommended starting point
# python train.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 5 --use_vic_loss 1 --lambda_v 1.0 --lambda_i 1.0 --lambda_c 0.04 --num_epoch 50 --learning_rate 1e-3 --n_episode 200

# Example 2 - 1-shot learning
# python train.py --method FSCT_cosine --dataset CUB --backbone ResNet34 --n_way 5 --k_shot 1 --use_vic_loss 1 --lambda_v 2.0 --lambda_i 1.0 --lambda_c 0.02 --num_epoch 50 --train_aug 1

# Example 3 - Standard baseline
# python train.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 5 --use_vic_loss 0 --num_epoch 50

# Example 4 - Custom weights
# python train.py --method FSCT_cosine --dataset CIFAR --backbone Conv4 --n_way 5 --k_shot 5 --use_vic_loss 1 --lambda_v 2.5 --lambda_i 1.0 --lambda_c 0.03 --num_epoch 50

# Example 5 - Full pipeline
# python train_test.py --method FSCT_cosine --dataset miniImagenet --backbone ResNet18 --n_way 5 --k_shot 5 --use_vic_loss 1 --lambda_v 1.0 --lambda_i 1.0 --lambda_c 0.04 --num_epoch 50 --test_iter 600 --wandb 1

# Example 6 - Memory optimized
# python train.py --method FSCT_cosine --dataset miniImagenet --backbone Conv6 --n_way 5 --k_shot 5 --use_vic_loss 1 --lambda_v 1.0 --lambda_i 1.0 --lambda_c 0.04 --num_epoch 50 --n_episode 150 --n_query 12
