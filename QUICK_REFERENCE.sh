#!/bin/bash
# Quick Reference Card for Unified Experiment Runner
# Usage: source this file or copy commands as needed

# ============================================
# BASIC USAGE
# ============================================

# Run ALL experiments (training, testing, ablation, qualitative, mcnemar, feature analysis)
python run_experiments.py --dataset miniImagenet --backbone Conv4 --n_way 5 --k_shot 1 --run_mode all

# ============================================
# RUN SPECIFIC MODES
# ============================================

# 1. Training and Testing ONLY
python run_experiments.py --dataset miniImagenet --backbone Conv4 --run_mode train_test

# 2. Ablation Study ONLY
python run_experiments.py --dataset miniImagenet --backbone Conv4 --run_mode ablation

# 3. Qualitative Analysis ONLY (t-SNE, confusion matrices)
python run_experiments.py --dataset miniImagenet --backbone Conv4 --run_mode qualitative

# 4. Feature Analysis ONLY (collapse detection, variance, correlation)
python run_experiments.py --dataset miniImagenet --backbone Conv4 --run_mode feature_analysis

# 5. McNemar's Test ONLY (statistical significance)
python run_experiments.py --dataset miniImagenet --backbone Conv4 --run_mode mcnemar

# ============================================
# DATASET OPTIONS
# ============================================

# miniImagenet (84x84 images)
python run_experiments.py --dataset miniImagenet --backbone Conv4

# CUB Birds (84x84 images)
python run_experiments.py --dataset CUB --backbone ResNet18

# CIFAR-FS (32x32 images)
python run_experiments.py --dataset CIFAR --backbone Conv4

# Omniglot (28x28 images, grayscale)
python run_experiments.py --dataset Omniglot --backbone Conv4

# ============================================
# BACKBONE OPTIONS
# ============================================

# Conv4 (lightweight, 4 convolutional blocks)
python run_experiments.py --backbone Conv4

# Conv6 (6 convolutional blocks)
python run_experiments.py --backbone Conv6

# ResNet18 (deeper, more parameters)
python run_experiments.py --backbone ResNet18

# ResNet34 (deepest option)
python run_experiments.py --backbone ResNet34

# ============================================
# FEW-SHOT SETTINGS
# ============================================

# 5-way 1-shot (hardest)
python run_experiments.py --n_way 5 --k_shot 1

# 5-way 5-shot (standard)
python run_experiments.py --n_way 5 --k_shot 5

# 10-way 1-shot (more classes)
python run_experiments.py --n_way 10 --k_shot 1

# Change query samples per class
python run_experiments.py --n_query 16  # default is 16

# ============================================
# TRAINING SETTINGS
# ============================================

# Custom epochs
python run_experiments.py --num_epochs 100

# Custom learning rate
python run_experiments.py --learning_rate 0.001

# Custom weight decay
python run_experiments.py --weight_decay 0.0001

# Different optimizer
python run_experiments.py --optimization Adam
python run_experiments.py --optimization AdamW  # default
python run_experiments.py --optimization SGD

# ============================================
# TESTING SETTINGS
# ============================================

# More test episodes (better statistics)
python run_experiments.py --test_iter 1000

# Fewer test episodes (faster, for debugging)
python run_experiments.py --test_iter 100

# ============================================
# ABLATION STUDY OPTIONS
# ============================================

# Run specific ablation experiments only
python run_experiments.py --run_mode ablation --ablation_experiments E1,E2,E6

# Run all 8 ablation experiments (default)
python run_experiments.py --run_mode ablation
# E1 = Full model
# E2 = Invariance + Dynamic
# E3 = Invariance + Covariance + Dynamic
# E4 = Invariance + Variance + Dynamic
# E5 = Full without Dynamic
# E6 = Baseline
# E7 = Covariance + Dynamic
# E8 = Variance + Dynamic

# ============================================
# USING PRE-TRAINED CHECKPOINTS
# ============================================

# Skip training, use pre-trained models
python run_experiments.py \
  --baseline_checkpoint ./checkpoints/baseline_best.tar \
  --proposed_checkpoint ./checkpoints/proposed_best.tar \
  --run_mode qualitative

# ============================================
# OUTPUT SETTINGS
# ============================================

# Custom output directory
python run_experiments.py --output_dir ./my_results

# Custom random seed
python run_experiments.py --seed 42

# ============================================
# QUICK DEBUGGING
# ============================================

# Fast run for testing (fewer epochs and iterations)
python run_experiments.py \
  --num_epochs 5 \
  --test_iter 50 \
  --run_mode train_test

# ============================================
# COMMON COMBINATIONS
# ============================================

# Full experiment on miniImagenet 5-way 1-shot
python run_experiments.py \
  --dataset miniImagenet \
  --backbone Conv4 \
  --n_way 5 \
  --k_shot 1 \
  --num_epochs 50 \
  --test_iter 600 \
  --run_mode all \
  --output_dir ./results/mini_5w1s

# Full experiment on CUB 5-way 5-shot with ResNet18
python run_experiments.py \
  --dataset CUB \
  --backbone ResNet18 \
  --n_way 5 \
  --k_shot 5 \
  --num_epochs 50 \
  --test_iter 600 \
  --run_mode all \
  --output_dir ./results/cub_5w5s

# Ablation study only on CIFAR
python run_experiments.py \
  --dataset CIFAR \
  --backbone Conv4 \
  --n_way 5 \
  --k_shot 1 \
  --run_mode ablation \
  --output_dir ./results/cifar_ablation

# ============================================
# OUTPUT STRUCTURE
# ============================================
# Results are saved in:
# {output_dir}/{dataset}_{backbone}_{n_way}w{k_shot}s/
#   ├── quantitative/          # Metrics and checkpoints
#   ├── qualitative/           # Visualizations
#   ├── ablation/              # Ablation study results
#   ├── mcnemar/               # Statistical tests
#   └── feature_analysis/      # Feature collapse analysis

# ============================================
# VIEWING RESULTS
# ============================================

# View JSON results
cat results/miniImagenet_Conv4_5w1s/quantitative/comparison_metrics.json

# View ablation results
cat results/miniImagenet_Conv4_5w1s/ablation/ablation_results.json

# View McNemar's test results
cat results/miniImagenet_Conv4_5w1s/mcnemar/significance_tests.json

# View feature analysis
cat results/miniImagenet_Conv4_5w1s/feature_analysis/feature_collapse_metrics.json

# Open visualizations (Linux)
xdg-open results/miniImagenet_Conv4_5w1s/qualitative/tsne_baseline.png

# Open visualizations (macOS)
open results/miniImagenet_Conv4_5w1s/qualitative/confusion_matrix_proposed.png

# ============================================
# GETTING HELP
# ============================================

# Show all options
python run_experiments.py --help

# Read documentation
cat UNIFIED_RUNNER_README.md

# View examples
bash examples_unified_runner.sh

# ============================================
# COMMON ISSUES
# ============================================

# Out of memory? Reduce test iterations
python run_experiments.py --test_iter 100

# Missing dependencies? Install requirements
pip install -r requirements.txt

# Dataset not found? Check configs.py data_dir paths
cat configs.py

echo "Quick reference loaded! Use these commands to run experiments."
