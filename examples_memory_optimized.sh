#!/bin/bash

# Example script demonstrating memory optimization features
# This script shows different configurations for various GPU memory constraints

echo "========================================="
echo "Memory-Optimized Training Examples"
echo "========================================="
echo ""

# Example 1: Small GPU (8GB or less)
echo "Example 1: Configuration for small GPUs (8GB)"
echo "---------------------------------------------"
cat << 'EOF'
python train_test.py \
  --method FSCT_cosine \
  --dataset miniImagenet \
  --backbone ResNet18 \
  --gradient_accumulation_steps 4 \
  --use_amp 1 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 8 \
  --num_epoch 50 \
  --wandb 0
EOF
echo ""

# Example 2: Medium GPU (16GB)
echo "Example 2: Configuration for medium GPUs (16GB)"
echo "------------------------------------------------"
cat << 'EOF'
python train_test.py \
  --method ProFOCT_cosine \
  --dataset miniImagenet \
  --backbone ResNet34 \
  --gradient_accumulation_steps 2 \
  --use_amp 1 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 16 \
  --dynamic_vic 1 \
  --num_epoch 50 \
  --wandb 0
EOF
echo ""

# Example 3: Large GPU (24GB+)
echo "Example 3: Configuration for large GPUs (24GB+)"
echo "------------------------------------------------"
cat << 'EOF'
python train_test.py \
  --method ProFOCT_cosine \
  --dataset miniImagenet \
  --backbone ResNet34 \
  --FETI 1 \
  --gradient_accumulation_steps 1 \
  --use_amp 1 \
  --n_way 5 \
  --k_shot 5 \
  --n_query 16 \
  --dynamic_vic 1 \
  --use_vic_on_attention 1 \
  --num_epoch 50 \
  --wandb 1
EOF
echo ""

# Example 4: Testing without CUDA (CPU)
echo "Example 4: Configuration for CPU (testing only)"
echo "------------------------------------------------"
cat << 'EOF'
python train_test.py \
  --method FSCT_cosine \
  --dataset miniImagenet \
  --backbone Conv4 \
  --gradient_accumulation_steps 1 \
  --use_amp 0 \
  --n_way 5 \
  --k_shot 1 \
  --n_query 5 \
  --n_episode 10 \
  --num_epoch 2 \
  --wandb 0
EOF
echo ""

echo "========================================="
echo "To run any example, copy the command and execute it"
echo "For more details, see MEMORY_OPTIMIZATION.md"
echo "========================================="
