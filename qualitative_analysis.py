"""
Qualitative Analysis for Enhanced Few-Shot Learning

This script performs qualitative analysis including:
- t-SNE visualization of feature embeddings
- Embedding space visualization
- Class separation analysis
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backbone
import configs
from data.datamgr import SetDataManager
from io_utils import model_dict, parse_args
from models.optimal_fewshot_enhanced import EnhancedOptimalFewShot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_features(model, data_loader, max_episodes=100):
    """
    Extract features from support and query sets.
    
    Args:
        model: Trained model
        data_loader: DataLoader for episodes
        max_episodes: Maximum number of episodes to process
    
    Returns:
        features: Concatenated features
        labels: Corresponding labels
        episode_ids: Episode IDs for each sample
    """
    model.eval()
    
    all_features = []
    all_labels = []
    all_episode_ids = []
    
    with torch.no_grad():
        for episode_idx, (x, _) in enumerate(tqdm.tqdm(data_loader, desc='Extracting features')):
            if episode_idx >= max_episodes:
                break
            
            x = x.to(device)
            
            # Parse features
            z_support, z_query = model.parse_feature(x, is_feature=False)
            
            # Reshape and extract
            N_support = z_support.size(0) * z_support.size(1)
            N_query = z_query.size(0) * z_query.size(1)
            
            z_support = z_support.contiguous().reshape(N_support, -1)
            z_query = z_query.contiguous().reshape(N_query, -1)
            
            # Project to transformer dimension
            support_features = model.projection(z_support)
            query_features = model.projection(z_query)
            
            # Apply invariance modules (if in eval mode, won't apply augmentation)
            support_features = model._apply_invariance_modules(support_features, is_training=False)
            query_features = model._apply_invariance_modules(query_features, is_training=False)
            
            # Combine support and query
            all_feats = torch.cat([support_features, query_features], dim=0)
            
            # Create labels (support then query)
            n_way = z_support.size(0)
            k_shot = z_support.size(1)
            n_query_per_class = z_query.size(1)
            
            support_labels = np.repeat(range(n_way), k_shot)
            query_labels = np.repeat(range(n_way), n_query_per_class)
            episode_labels = np.concatenate([support_labels, query_labels])
            
            # Episode IDs
            episode_ids = np.ones(len(episode_labels)) * episode_idx
            
            all_features.append(all_feats.cpu().numpy())
            all_labels.append(episode_labels)
            all_episode_ids.append(episode_ids)
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    episode_ids = np.concatenate(all_episode_ids, axis=0)
    
    return features, labels, episode_ids


def plot_tsne(features, labels, episode_ids, save_path, title='t-SNE Visualization', 
              n_components=2, perplexity=30):
    """
    Create t-SNE visualization of features.
    
    Args:
        features: Feature array [N, D]
        labels: Class labels [N]
        episode_ids: Episode IDs [N]
        save_path: Path to save figure
        title: Plot title
        n_components: Number of t-SNE dimensions (2 or 3)
        perplexity: t-SNE perplexity parameter
    """
    print(f"\nComputing t-SNE with perplexity={perplexity}...")
    
    # Run t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, 
                n_iter=1000, verbose=1)
    features_tsne = tsne.fit_transform(features)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    if n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                           c=labels, cmap='tab10', alpha=0.6, s=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    else:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2],
                           c=labels, cmap='tab10', alpha=0.6, s=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_zlabel('t-SNE Dimension 3', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()


def plot_pca(features, labels, save_path, title='PCA Visualization'):
    """
    Create PCA visualization of features.
    
    Args:
        features: Feature array [N, D]
        labels: Class labels [N]
        save_path: Path to save figure
        title: Plot title
    """
    print("\nComputing PCA...")
    
    # Run PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], 
                        c=labels, cmap='tab10', alpha=0.6, s=20)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved PCA plot to {save_path}")
    plt.close()


def plot_embedding_space(features, labels, save_path, title='Embedding Space Analysis'):
    """
    Create multiple views of the embedding space.
    
    Args:
        features: Feature array [N, D]
        labels: Class labels [N]
        save_path: Path to save figure
        title: Plot title
    """
    print("\nCreating embedding space analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # 1. Class-wise feature distribution (first 2 PCs)
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    ax = axes[0, 0]
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                  label=f'Class {label}', alpha=0.6, s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Feature magnitude distribution per class
    ax = axes[0, 1]
    feature_norms = np.linalg.norm(features, axis=1)
    for label in unique_labels:
        mask = labels == label
        ax.hist(feature_norms[mask], alpha=0.5, bins=30, label=f'Class {label}')
    ax.set_xlabel('Feature Norm')
    ax.set_ylabel('Frequency')
    ax.set_title('Feature Magnitude Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Class separation (pairwise distances)
    ax = axes[1, 0]
    class_centers = []
    for label in unique_labels:
        mask = labels == label
        center = features[mask].mean(axis=0)
        class_centers.append(center)
    class_centers = np.array(class_centers)
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(class_centers, metric='euclidean'))
    
    im = ax.imshow(distances, cmap='viridis')
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    ax.set_title('Inter-Class Distance Matrix')
    plt.colorbar(im, ax=ax)
    
    # 4. Intra-class variance
    ax = axes[1, 1]
    variances = []
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        variance = np.var(class_features, axis=0).mean()
        variances.append(variance)
    
    ax.bar(unique_labels, variances, alpha=0.7)
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Intra-Class Variance')
    ax.set_title('Intra-Class Variance per Class')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved embedding space analysis to {save_path}")
    plt.close()


def main():
    """Main function for qualitative analysis"""
    parser = argparse.ArgumentParser(description='Qualitative Analysis for Enhanced Few-Shot Learning')
    
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
    
    # Analysis parameters
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--n_episodes', default=100, type=int,
                       help='Number of episodes to analyze')
    parser.add_argument('--output_dir', default='./qualitative_results',
                       help='Output directory for plots')
    parser.add_argument('--perplexity', default=30, type=int,
                       help='t-SNE perplexity parameter')
    
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
    print("Qualitative Analysis for Enhanced Few-Shot Learning")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Task: {args.n_way}-way {args.k_shot}-shot")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    
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
    
    # Extract features
    features, labels, episode_ids = extract_features(model, data_loader, max_episodes=args.n_episodes)
    
    print(f"\nExtracted {features.shape[0]} feature vectors of dimension {features.shape[1]}")
    print(f"Number of unique classes: {len(np.unique(labels))}")
    print(f"Number of episodes: {len(np.unique(episode_ids))}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    # 1. t-SNE 2D
    plot_tsne(
        features, labels, episode_ids,
        os.path.join(args.output_dir, 'tsne_2d.png'),
        title=f't-SNE 2D - {args.dataset} ({args.n_way}-way {args.k_shot}-shot)',
        n_components=2,
        perplexity=args.perplexity
    )
    
    # 2. t-SNE 3D
    plot_tsne(
        features, labels, episode_ids,
        os.path.join(args.output_dir, 'tsne_3d.png'),
        title=f't-SNE 3D - {args.dataset} ({args.n_way}-way {args.k_shot}-shot)',
        n_components=3,
        perplexity=args.perplexity
    )
    
    # 3. PCA
    plot_pca(
        features, labels,
        os.path.join(args.output_dir, 'pca.png'),
        title=f'PCA - {args.dataset} ({args.n_way}-way {args.k_shot}-shot)'
    )
    
    # 4. Embedding space analysis
    plot_embedding_space(
        features, labels,
        os.path.join(args.output_dir, 'embedding_analysis.png'),
        title=f'Embedding Space Analysis - {args.dataset} ({args.n_way}-way {args.k_shot}-shot)'
    )
    
    print("\n" + "=" * 80)
    print("Qualitative Analysis Complete!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - tsne_2d.png")
    print(f"  - tsne_3d.png")
    print(f"  - pca.png")
    print(f"  - embedding_analysis.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
