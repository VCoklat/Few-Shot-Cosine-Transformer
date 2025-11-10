#!/usr/bin/env python
"""
Example training script for EnhancedFSCT

This script demonstrates how to train the Enhanced Few-Shot Cosine Transformer
with VIC regularization and Mahalanobis distance classification.
"""

import torch
import argparse
import sys
from methods.enhanced_fsct import EnhancedFSCT
import backbone

def parse_args():
    parser = argparse.ArgumentParser(description='EnhancedFSCT Example')
    parser.add_argument('--backbone', default='Conv4', type=str,
                        help='Feature backbone: Conv4, Conv6, ResNet18, ResNet34')
    parser.add_argument('--dataset', default='miniImagenet', type=str,
                        help='Dataset: miniImagenet, CIFAR, CUB, etc.')
    parser.add_argument('--n_way', default=5, type=int,
                        help='Number of classes per episode')
    parser.add_argument('--k_shot', default=1, type=int,
                        help='Number of shots per class')
    parser.add_argument('--n_query', default=8, type=int,
                        help='Number of queries per class')
    parser.add_argument('--depth', default=2, type=int,
                        help='Number of cosine encoder blocks')
    parser.add_argument('--heads', default=4, type=int,
                        help='Number of attention heads')
    parser.add_argument('--dim_head', default=64, type=int,
                        help='Dimension per attention head')
    parser.add_argument('--mlp_dim', default=512, type=int,
                        help='FFN hidden dimension')
    parser.add_argument('--lambda_I', default=9.0, type=float,
                        help='Initial weight for Invariance loss')
    parser.add_argument('--lambda_V', default=0.5, type=float,
                        help='Initial weight for Variance loss')
    parser.add_argument('--lambda_C', default=0.5, type=float,
                        help='Initial weight for Covariance loss')
    parser.add_argument('--use_uncertainty', default=1, type=int,
                        help='Use uncertainty weighting (1) or not (0)')
    parser.add_argument('--use_gradnorm', default=0, type=int,
                        help='Use GradNorm (1) or not (0)')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Weight decay')
    parser.add_argument('--use_amp', default=1, type=int,
                        help='Use mixed precision training (1) or not (0)')
    parser.add_argument('--grad_clip', default=1.0, type=float,
                        help='Gradient clipping norm')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("Enhanced Few-Shot Cosine Transformer (EnhancedFSCT) Example")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Task: {args.n_way}-way {args.k_shot}-shot")
    print(f"  Queries: {args.n_query} per class")
    print(f"  Architecture: depth={args.depth}, heads={args.heads}, dim_head={args.dim_head}")
    print(f"  VIC Weights: λ_I={args.lambda_I}, λ_V={args.lambda_V}, λ_C={args.lambda_C}")
    print(f"  Dynamic Weighting: {'Uncertainty' if args.use_uncertainty else ('GradNorm' if args.use_gradnorm else 'Fixed')}")
    print(f"  Mixed Precision: {'Enabled' if args.use_amp else 'Disabled'}")
    print()
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Define feature extractor
    def feature_model():
        backbone_dict = {
            'Conv4': backbone.Conv4,
            'Conv6': backbone.Conv6,
            'ResNet18': backbone.ResNet18,
            'ResNet34': backbone.ResNet34,
        }
        
        if args.backbone not in backbone_dict:
            raise ValueError(f"Unknown backbone: {args.backbone}")
        
        backbone_fn = backbone_dict[args.backbone]
        
        if 'ResNet' in args.backbone:
            return backbone_fn(FETI=0, dataset=args.dataset, flatten=True)
        else:
            return backbone_fn(dataset=args.dataset, flatten=True)
    
    # Create model
    print("Initializing model...")
    model = EnhancedFSCT(
        feature_model,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        lambda_I=args.lambda_I,
        lambda_V=args.lambda_V,
        lambda_C=args.lambda_C,
        use_uncertainty_weighting=bool(args.use_uncertainty),
        use_gradnorm=bool(args.use_gradnorm),
        shrinkage_alpha=None,  # Adaptive
        epsilon=1e-4
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    print("Model initialized successfully!")
    print()
    
    # Create dummy episode for demonstration
    print("Testing with dummy episode...")
    image_size = 84 if 'Conv' in args.backbone else 224
    
    # Shape: (n_way, k_shot + n_query, C, H, W)
    dummy_episode = torch.randn(
        args.n_way,
        args.k_shot + args.n_query,
        3, image_size, image_size
    ).to(device)
    
    # Forward pass
    model.train()
    optimizer.zero_grad()
    
    if args.use_amp:
        with torch.cuda.amp.autocast():
            acc, loss = model.set_forward_loss(dummy_episode)
        
        scaler.scale(loss).backward()
        
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            print(f"Gradient norm: {grad_norm:.4f} (clipped to {args.grad_clip})")
        
        scaler.step(optimizer)
        scaler.update()
    else:
        acc, loss = model.set_forward_loss(dummy_episode)
        loss.backward()
        
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            print(f"Gradient norm: {grad_norm:.4f} (clipped to {args.grad_clip})")
        
        optimizer.step()
    
    print(f"✓ Training step successful!")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Loss: {loss.item():.4f}")
    print()
    
    # Test inference
    model.eval()
    with torch.no_grad():
        scores = model.set_forward(dummy_episode)
    
    print(f"✓ Inference successful!")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Score range: [{scores.min().item():.2f}, {scores.max().item():.2f}]")
    print()
    
    # Display learned weights if using uncertainty weighting
    if args.use_uncertainty:
        print("Learned uncertainty weights:")
        print(f"  log_var_I: {model.log_var_I.item():.4f} → λ_I ∝ {torch.exp(-model.log_var_I).item():.4f}")
        print(f"  log_var_V: {model.log_var_V.item():.4f} → λ_V ∝ {torch.exp(-model.log_var_V).item():.4f}")
        print(f"  log_var_C: {model.log_var_C.item():.4f} → λ_C ∝ {torch.exp(-model.log_var_C).item():.4f}")
        print()
    
    # Display prototype weights
    print("Sample prototype weights (class 0):")
    weights = torch.softmax(model.proto_weight[0], dim=0).squeeze()
    print(f"  {weights.cpu().tolist()}")
    print(f"  Sum: {weights.sum().item():.6f} (should be 1.0)")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print()
    print("To train on real data, use train_test.py with --method EnhancedFSCT")
    print("Example:")
    print(f"  python train_test.py --method EnhancedFSCT --dataset {args.dataset} \\")
    print(f"    --backbone {args.backbone} --n_way {args.n_way} --k_shot {args.k_shot} \\")
    print(f"    --use_amp 1 --grad_clip 1.0")
    print("=" * 70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
