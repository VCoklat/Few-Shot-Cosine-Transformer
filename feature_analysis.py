"""
Comprehensive Feature Space Analysis for Few-Shot Classification

This module provides 8 statistical evaluation metrics to quantify model performance 
uncertainty, feature quality, and component contributions in few-shot learning:

1. Feature Collapse Detection - identifies dimensions with std < 1e-4
2. Feature Utilization - percentile-based range vs theoretical maximum
3. Feature Diversity - coefficient of variation from class centroids
4. Feature Redundancy - Pearson correlation pairs + PCA effective dimensionality
5. Intra-class Consistency - Euclidean distance and cosine similarity
6. Confusing Pairs - inter-centroid distances ranked by proximity
7. Class Imbalance Ratio - minority/majority class sample counts
8. Statistical Confidence - 95% CI from per-episode accuracies

Author: dvh
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from scipy.stats import pearsonr
    from sklearn.decomposition import PCA
    from sklearn.metrics import f1_score, confusion_matrix
    from sklearn.manifold import TSNE
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy/sklearn not available. Some features will be limited.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None
    warnings.warn("matplotlib/seaborn not available. Visualizations will be disabled.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def detect_feature_collapse(features: np.ndarray, threshold: float = 1e-4) -> Dict:
    """
    Detects dimensions with very low variance (std < threshold).
    
    Args:
        features: Shape (n_samples, n_features)
        threshold: Variance threshold for collapse detection
        
    Returns:
        Dict with collapsed dimensions and statistics
    """
    feature_std = np.std(features, axis=0)
    collapsed_dims = np.where(feature_std < threshold)[0]
    
    return {
        'collapsed_dimensions': collapsed_dims.tolist(),
        'num_collapsed': len(collapsed_dims),
        'total_dimensions': features.shape[1],
        'collapse_ratio': len(collapsed_dims) / features.shape[1] if features.shape[1] > 0 else 0,
        'min_std': float(np.min(feature_std)),
        'mean_std': float(np.mean(feature_std)),
        'max_std': float(np.max(feature_std))
    }


def compute_feature_utilization(features: np.ndarray, 
                               lower_percentile: float = 5,
                               upper_percentile: float = 95) -> Dict:
    """
    Measures how well features use their available range using percentiles.
    
    Args:
        features: Shape (n_samples, n_features)
        lower_percentile: Lower percentile for range calculation
        upper_percentile: Upper percentile for range calculation
        
    Returns:
        Dict with utilization metrics
    """
    # Calculate percentile-based ranges
    lower = np.percentile(features, lower_percentile, axis=0)
    upper = np.percentile(features, upper_percentile, axis=0)
    percentile_range = upper - lower
    
    # Theoretical maximum (full data range)
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    theoretical_range = max_vals - min_vals
    
    # Avoid division by zero
    theoretical_range = np.maximum(theoretical_range, 1e-10)
    utilization = percentile_range / theoretical_range
    
    return {
        'mean_utilization': float(np.mean(utilization)),
        'median_utilization': float(np.median(utilization)),
        'min_utilization': float(np.min(utilization)),
        'max_utilization': float(np.max(utilization)),
        'std_utilization': float(np.std(utilization)),
        'percentile_range': {
            'mean': float(np.mean(percentile_range)),
            'std': float(np.std(percentile_range))
        },
        'theoretical_range': {
            'mean': float(np.mean(theoretical_range)),
            'std': float(np.std(theoretical_range))
        }
    }


def compute_feature_diversity(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Measures diversity using coefficient of variation of class centroids.
    
    Args:
        features: Shape (n_samples, n_features)
        labels: Shape (n_samples,)
        
    Returns:
        Dict with diversity metrics
    """
    unique_labels = np.unique(labels)
    
    # Compute class centroids
    centroids = []
    for label in unique_labels:
        class_features = features[labels == label]
        if len(class_features) > 0:
            centroids.append(np.mean(class_features, axis=0))
    
    centroids = np.array(centroids)
    
    if len(centroids) == 0:
        return {
            'coefficient_of_variation': 0.0,
            'centroid_mean': 0.0,
            'centroid_std': 0.0,
            'num_classes': 0
        }
    
    # Coefficient of variation per dimension
    centroid_mean = np.mean(centroids, axis=0)
    centroid_std = np.std(centroids, axis=0)
    
    # Avoid division by zero
    cv = np.zeros_like(centroid_mean)
    non_zero_mask = np.abs(centroid_mean) > 1e-10
    cv[non_zero_mask] = centroid_std[non_zero_mask] / np.abs(centroid_mean[non_zero_mask])
    
    return {
        'coefficient_of_variation': float(np.mean(cv)),
        'cv_std': float(np.std(cv)),
        'centroid_mean': float(np.mean(centroid_mean)),
        'centroid_std': float(np.mean(centroid_std)),
        'num_classes': len(unique_labels)
    }


def compute_feature_redundancy(features: np.ndarray, 
                               corr_threshold_high: float = 0.9,
                               corr_threshold_med: float = 0.7,
                               variance_threshold: float = 0.95) -> Dict:
    """
    Analyzes feature redundancy using correlation and PCA.
    
    Args:
        features: Shape (n_samples, n_features)
        corr_threshold_high: Threshold for high correlation pairs
        corr_threshold_med: Threshold for medium correlation pairs
        variance_threshold: Variance threshold for PCA effective dimensionality
        
    Returns:
        Dict with redundancy metrics
    """
    if not SCIPY_AVAILABLE:
        return {
            'high_correlation_pairs': 0,
            'medium_correlation_pairs': 0,
            'mean_abs_correlation': 0.0,
            'effective_dimensions_95pct': features.shape[1],
            'variance_explained_95pct': 0.95,
            'note': 'scipy not available'
        }
    
    n_features = features.shape[1]
    n_samples = features.shape[0]
    
    # Initialize correlation stats
    high_corr_pairs = 0
    med_corr_pairs = 0
    all_corr = []
    
    # Compute pairwise correlations (sample efficiently for large feature spaces)
    if n_features <= 100 and n_samples > 1:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    corr, _ = pearsonr(features[:, i], features[:, j])
                    if not np.isnan(corr):
                        all_corr.append(abs(corr))
                        if abs(corr) > corr_threshold_high:
                            high_corr_pairs += 1
                        elif abs(corr) > corr_threshold_med:
                            med_corr_pairs += 1
                except:
                    pass
    
    mean_corr = float(np.mean(all_corr)) if all_corr else 0.0
    
    # PCA for effective dimensionality
    effective_dims = n_features
    variance_explained = variance_threshold
    
    try:
        if n_samples > 1 and n_features > 1:
            n_components = min(n_samples - 1, n_features)
            pca = PCA(n_components=n_components)
            pca.fit(features)
            
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            effective_dims = int(np.argmax(cumsum_variance >= variance_threshold) + 1)
            variance_explained = float(cumsum_variance[effective_dims - 1]) if effective_dims > 0 else variance_threshold
    except:
        pass
    
    return {
        'high_correlation_pairs': int(high_corr_pairs),
        'medium_correlation_pairs': int(med_corr_pairs),
        'mean_abs_correlation': float(mean_corr),
        'effective_dimensions_95pct': int(effective_dims),
        'variance_explained_95pct': float(variance_explained),
        'total_dimensions': int(n_features),
        'dimensionality_reduction_ratio': float(effective_dims / n_features) if n_features > 0 else 1.0
    }


def compute_intraclass_consistency(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Measures intra-class consistency using distance and cosine similarity.
    
    Args:
        features: Shape (n_samples, n_features)
        labels: Shape (n_samples,)
        
    Returns:
        Dict with consistency metrics
    """
    unique_labels = np.unique(labels)
    
    euclidean_distances = []
    cosine_similarities = []
    
    for label in unique_labels:
        class_features = features[labels == label]
        
        if len(class_features) < 2:
            continue
        
        # Compute centroid
        centroid = np.mean(class_features, axis=0)
        
        # Euclidean distances to centroid
        dists = np.linalg.norm(class_features - centroid, axis=1)
        euclidean_distances.extend(dists.tolist())
        
        # Cosine similarities to centroid
        for feat in class_features:
            # Avoid division by zero
            norm_feat = np.linalg.norm(feat)
            norm_centroid = np.linalg.norm(centroid)
            
            if norm_feat > 1e-10 and norm_centroid > 1e-10:
                cos_sim = np.dot(feat, centroid) / (norm_feat * norm_centroid)
                cosine_similarities.append(cos_sim)
    
    return {
        'mean_euclidean_distance': float(np.mean(euclidean_distances)) if euclidean_distances else 0.0,
        'std_euclidean_distance': float(np.std(euclidean_distances)) if euclidean_distances else 0.0,
        'mean_cosine_similarity': float(np.mean(cosine_similarities)) if cosine_similarities else 1.0,
        'std_cosine_similarity': float(np.std(cosine_similarities)) if cosine_similarities else 0.0,
        'min_cosine_similarity': float(np.min(cosine_similarities)) if cosine_similarities else 1.0,
        'num_classes_evaluated': len(unique_labels)
    }


def identify_confusing_pairs(features: np.ndarray, labels: np.ndarray, top_k: int = 5) -> Dict:
    """
    Identifies most confusing class pairs based on centroid distances.
    
    Args:
        features: Shape (n_samples, n_features)
        labels: Shape (n_samples,)
        top_k: Number of top confusing pairs to return
        
    Returns:
        Dict with confusing pairs information
    """
    unique_labels = np.unique(labels)
    
    # Compute class centroids
    centroids = {}
    for label in unique_labels:
        class_features = features[labels == label]
        if len(class_features) > 0:
            centroids[int(label)] = np.mean(class_features, axis=0)
    
    # Compute pairwise distances between centroids
    pairs_distances = []
    label_list = list(centroids.keys())
    
    for i, label1 in enumerate(label_list):
        for label2 in label_list[i+1:]:
            dist = np.linalg.norm(centroids[label1] - centroids[label2])
            pairs_distances.append({
                'class_1': int(label1),
                'class_2': int(label2),
                'distance': float(dist)
            })
    
    # Sort by distance (ascending - closest pairs are most confusing)
    pairs_distances.sort(key=lambda x: x['distance'])
    
    return {
        'top_confusing_pairs': pairs_distances[:top_k],
        'mean_inter_centroid_distance': float(np.mean([p['distance'] for p in pairs_distances])) if pairs_distances else 0.0,
        'std_inter_centroid_distance': float(np.std([p['distance'] for p in pairs_distances])) if pairs_distances else 0.0,
        'min_inter_centroid_distance': float(pairs_distances[0]['distance']) if pairs_distances else 0.0,
        'max_inter_centroid_distance': float(pairs_distances[-1]['distance']) if pairs_distances else 0.0,
        'num_class_pairs': len(pairs_distances)
    }


def compute_class_imbalance(labels: np.ndarray) -> Dict:
    """
    Computes class imbalance ratio.
    
    Args:
        labels: Shape (n_samples,)
        
    Returns:
        Dict with imbalance metrics
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if len(counts) == 0:
        return {
            'imbalance_ratio': 1.0,
            'min_samples': 0,
            'max_samples': 0,
            'mean_samples': 0.0,
            'std_samples': 0.0
        }
    
    min_count = np.min(counts)
    max_count = np.max(counts)
    
    imbalance_ratio = min_count / max_count if max_count > 0 else 1.0
    
    return {
        'imbalance_ratio': float(imbalance_ratio),
        'min_samples': int(min_count),
        'max_samples': int(max_count),
        'mean_samples': float(np.mean(counts)),
        'std_samples': float(np.std(counts)),
        'num_classes': len(unique_labels)
    }


def compute_statistical_confidence(episode_accuracies: np.ndarray,
                                   predictions: Optional[np.ndarray] = None,
                                   labels: Optional[np.ndarray] = None) -> Dict:
    """
    Computes 95% confidence intervals and per-class F1 scores.
    
    Args:
        episode_accuracies: Array of per-episode accuracies
        predictions: Optional predictions array
        labels: Optional true labels array
        
    Returns:
        Dict with statistical confidence metrics
    """
    if len(episode_accuracies) == 0:
        return {
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'confidence_interval_95': [0.0, 0.0],
            'z_score': 1.96
        }
    
    mean_acc = np.mean(episode_accuracies)
    std_acc = np.std(episode_accuracies)
    n = len(episode_accuracies)
    
    # 95% confidence interval using z-score approximation
    z_score = 1.96
    margin_of_error = z_score * std_acc / np.sqrt(n)
    
    result = {
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'confidence_interval_95': [
            float(mean_acc - margin_of_error),
            float(mean_acc + margin_of_error)
        ],
        'margin_of_error': float(margin_of_error),
        'z_score': float(z_score),
        'num_episodes': int(n)
    }
    
    # Add per-class F1 scores if predictions and labels are provided
    if predictions is not None and labels is not None and SCIPY_AVAILABLE:
        try:
            class_f1 = f1_score(labels, predictions, average=None)
            macro_f1 = f1_score(labels, predictions, average='macro')
            conf_mat = confusion_matrix(labels, predictions)
            
            result.update({
                'per_class_f1': class_f1.tolist(),
                'macro_f1': float(macro_f1),
                'confusion_matrix': conf_mat.tolist()
            })
        except:
            pass
    
    return result


def comprehensive_feature_analysis(features: np.ndarray, 
                                   labels: np.ndarray,
                                   episode_accuracies: Optional[np.ndarray] = None,
                                   predictions: Optional[np.ndarray] = None) -> Dict:
    """
    Runs all 8 feature analysis metrics in one go.
    
    Args:
        features: Shape (n_samples, n_features)
        labels: Shape (n_samples,) - true labels
        episode_accuracies: Optional array of per-episode accuracies
        predictions: Optional predictions for statistical analysis
        
    Returns:
        Dict containing all 8 metric results
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE FEATURE SPACE ANALYSIS")
    print("="*60)
    
    results = {}
    
    # 1. Feature Collapse Detection
    print("\n[1/8] Detecting feature collapse...")
    results['feature_collapse'] = detect_feature_collapse(features)
    
    # 2. Feature Utilization
    print("[2/8] Computing feature utilization...")
    results['feature_utilization'] = compute_feature_utilization(features)
    
    # 3. Feature Diversity
    print("[3/8] Analyzing feature diversity...")
    results['feature_diversity'] = compute_feature_diversity(features, labels)
    
    # 4. Feature Redundancy
    print("[4/8] Assessing feature redundancy...")
    results['feature_redundancy'] = compute_feature_redundancy(features)
    
    # 5. Intra-class Consistency
    print("[5/8] Evaluating intra-class consistency...")
    results['intraclass_consistency'] = compute_intraclass_consistency(features, labels)
    
    # 6. Confusing Pairs
    print("[6/8] Identifying confusing class pairs...")
    results['confusing_pairs'] = identify_confusing_pairs(features, labels)
    
    # 7. Class Imbalance
    print("[7/8] Computing class imbalance...")
    results['class_imbalance'] = compute_class_imbalance(labels)
    
    # 8. Statistical Confidence
    print("[8/8] Calculating statistical confidence...")
    if episode_accuracies is not None:
        results['statistical_confidence'] = compute_statistical_confidence(
            episode_accuracies, predictions, labels
        )
    else:
        results['statistical_confidence'] = {
            'note': 'Episode accuracies not provided'
        }
    
    print("\n✓ Analysis complete!")
    
    return results


def print_feature_analysis_summary(results: Dict) -> None:
    """
    Pretty print summary of feature analysis results.
    
    Args:
        results: Output from comprehensive_feature_analysis()
    """
    print("\n" + "="*60)
    print("FEATURE ANALYSIS SUMMARY")
    print("="*60)
    
    # 1. Feature Collapse
    if 'feature_collapse' in results:
        fc = results['feature_collapse']
        print(f"\n📉 Feature Collapse:")
        print(f"  Collapsed dimensions: {fc['num_collapsed']}/{fc['total_dimensions']} ({fc['collapse_ratio']*100:.1f}%)")
        print(f"  Min/Mean/Max std: {fc['min_std']:.6f} / {fc['mean_std']:.6f} / {fc['max_std']:.6f}")
    
    # 2. Feature Utilization
    if 'feature_utilization' in results:
        fu = results['feature_utilization']
        print(f"\n📊 Feature Utilization:")
        print(f"  Mean: {fu['mean_utilization']:.4f}")
        print(f"  Median: {fu['median_utilization']:.4f}")
        print(f"  Range: [{fu['min_utilization']:.4f}, {fu['max_utilization']:.4f}]")
    
    # 3. Feature Diversity
    if 'feature_diversity' in results:
        fd = results['feature_diversity']
        print(f"\n🎨 Feature Diversity:")
        print(f"  Coefficient of Variation: {fd['coefficient_of_variation']:.4f}")
        print(f"  Num classes: {fd['num_classes']}")
    
    # 4. Feature Redundancy
    if 'feature_redundancy' in results:
        fr = results['feature_redundancy']
        print(f"\n🔄 Feature Redundancy:")
        print(f"  High correlation pairs (>0.9): {fr['high_correlation_pairs']}")
        print(f"  Medium correlation pairs (>0.7): {fr['medium_correlation_pairs']}")
        print(f"  Effective dims (95% variance): {fr['effective_dimensions_95pct']}/{fr.get('total_dimensions', 'N/A')}")
    
    # 5. Intra-class Consistency
    if 'intraclass_consistency' in results:
        ic = results['intraclass_consistency']
        print(f"\n🎯 Intra-class Consistency:")
        print(f"  Mean Euclidean distance: {ic['mean_euclidean_distance']:.4f} ± {ic['std_euclidean_distance']:.4f}")
        print(f"  Mean Cosine similarity: {ic['mean_cosine_similarity']:.4f} ± {ic['std_cosine_similarity']:.4f}")
    
    # 6. Confusing Pairs
    if 'confusing_pairs' in results:
        cp = results['confusing_pairs']
        print(f"\n🤔 Most Confusing Class Pairs:")
        for i, pair in enumerate(cp['top_confusing_pairs'][:3], 1):
            print(f"  {i}. Classes {pair['class_1']} ↔ {pair['class_2']}: distance = {pair['distance']:.4f}")
    
    # 7. Class Imbalance
    if 'class_imbalance' in results:
        ci = results['class_imbalance']
        print(f"\n⚖️ Class Imbalance:")
        print(f"  Imbalance ratio: {ci['imbalance_ratio']:.4f}")
        print(f"  Samples per class: {ci['min_samples']} - {ci['max_samples']} (mean: {ci['mean_samples']:.1f})")
    
    # 8. Statistical Confidence
    if 'statistical_confidence' in results:
        sc = results['statistical_confidence']
        if 'mean_accuracy' in sc:
            print(f"\n📈 Statistical Confidence:")
            print(f"  Mean accuracy: {sc['mean_accuracy']*100:.2f}%")
            print(f"  95% CI: [{sc['confidence_interval_95'][0]*100:.2f}%, {sc['confidence_interval_95'][1]*100:.2f}%]")
            if 'macro_f1' in sc:
                print(f"  Macro F1: {sc['macro_f1']:.4f}")
    
    print("\n" + "="*60)


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def visualize_embedding_space(features: np.ndarray, 
                              labels: np.ndarray,
                              method: str = 'tsne',
                              n_components: int = 2,
                              save_path: Optional[str] = None,
                              title: Optional[str] = None,
                              show: bool = True,
                              **kwargs) -> Optional[Figure]:
    """
    Visualize feature embeddings in 2D or 3D space using dimensionality reduction.
    
    Args:
        features: Shape (n_samples, n_features)
        labels: Shape (n_samples,)
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        n_components: Number of dimensions (2 or 3)
        save_path: Path to save the visualization (e.g., 'embedding_space.png')
        title: Custom title for the plot
        show: Whether to display the plot (default True)
        **kwargs: Additional arguments for the reduction method
        
    Returns:
        matplotlib Figure object or None if visualization unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("matplotlib not available. Cannot create visualizations.")
        return None
    
    if not SCIPY_AVAILABLE and method != 'pca':
        warnings.warn(f"scipy not available. Cannot use {method} method.")
        return None
    
    print(f"\n🎨 Visualizing embedding space using {method.upper()}...")
    
    # Perform dimensionality reduction
    if method == 'tsne':
        if not SCIPY_AVAILABLE:
            return None
        from sklearn.manifold import TSNE
        perplexity = kwargs.get('perplexity', min(30, len(features) - 1))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        embeddings = reducer.fit_transform(features)
    elif method == 'pca':
        if not SCIPY_AVAILABLE:
            warnings.warn("sklearn not available. Cannot use PCA.")
            return None
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
        embeddings = reducer.fit_transform(features)
        variance_explained = reducer.explained_variance_ratio_
        print(f"  Variance explained: {variance_explained.sum()*100:.2f}%")
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            warnings.warn("umap-learn not available. Using PCA instead.")
            if not SCIPY_AVAILABLE:
                return None
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
            embeddings = reducer.fit_transform(features)
        else:
            n_neighbors = kwargs.get('n_neighbors', min(15, len(features) - 1))
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
            embeddings = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne', 'pca', or 'umap'.")
    
    # Create visualization
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                           c=labels, cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        plt.colorbar(scatter, ax=ax, label='Class')
    else:  # 3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                           c=labels, cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_zlabel(f'{method.upper()} Component 3')
        plt.colorbar(scatter, ax=ax, label='Class')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Feature Embedding Space ({method.upper()})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_attention_maps(attention_weights: np.ndarray,
                            save_path: Optional[str] = None,
                            title: Optional[str] = None,
                            query_labels: Optional[np.ndarray] = None,
                            support_labels: Optional[np.ndarray] = None,
                            show: bool = True) -> Optional[Figure]:
    """
    Visualize attention weight matrices as heatmaps.
    
    Args:
        attention_weights: Shape (n_queries, n_support) or (n_heads, n_queries, n_support)
        save_path: Path to save the visualization (e.g., 'attention_maps.png')
        title: Custom title for the plot
        query_labels: Optional labels for query samples
        support_labels: Optional labels for support samples
        show: Whether to display the plot (default True)
        
    Returns:
        matplotlib Figure object or None if visualization unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("matplotlib not available. Cannot create visualizations.")
        return None
    
    print("\n🔍 Visualizing attention maps...")
    
    # Handle multi-head attention
    if len(attention_weights.shape) == 3:
        n_heads = attention_weights.shape[0]
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_heads == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            sns.heatmap(attention_weights[head_idx], 
                       cmap='viridis', annot=False, 
                       cbar=True, ax=ax,
                       xticklabels=support_labels if support_labels is not None else False,
                       yticklabels=query_labels if query_labels is not None else False)
            ax.set_title(f'Attention Head {head_idx + 1}')
            ax.set_xlabel('Support Samples')
            ax.set_ylabel('Query Samples')
        
        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].axis('off')
    else:
        # Single attention map
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_weights, 
                   cmap='viridis', annot=False, 
                   cbar=True, ax=ax,
                   xticklabels=support_labels if support_labels is not None else False,
                   yticklabels=query_labels if query_labels is not None else False)
        ax.set_xlabel('Support Samples')
        ax.set_ylabel('Query Samples')
    
    if title:
        plt.suptitle(title)
    else:
        plt.suptitle('Attention Weight Maps')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_weight_distributions(model_weights: Dict[str, np.ndarray],
                                   save_path: Optional[str] = None,
                                   title: Optional[str] = None,
                                   layer_names: Optional[List[str]] = None,
                                   show: bool = True) -> Optional[Figure]:
    """
    Visualize distribution of model weights across layers.
    
    Args:
        model_weights: Dictionary mapping layer names to weight arrays
        save_path: Path to save the visualization (e.g., 'weight_distributions.png')
        title: Custom title for the plot
        layer_names: Optional list of specific layer names to visualize
        show: Whether to display the plot (default True)
        
    Returns:
        matplotlib Figure object or None if visualization unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("matplotlib not available. Cannot create visualizations.")
        return None
    
    print("\n📊 Visualizing weight distributions...")
    
    # Filter layers if specified
    if layer_names:
        model_weights = {k: v for k, v in model_weights.items() if k in layer_names}
    
    if not model_weights:
        warnings.warn("No weights to visualize.")
        return None
    
    n_layers = len(model_weights)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (layer_name, weights) in enumerate(model_weights.items()):
        ax = axes[idx]
        
        # Flatten weights
        weights_flat = weights.flatten()
        
        # Plot histogram
        ax.hist(weights_flat, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{layer_name}\nMean: {weights_flat.mean():.4f}, Std: {weights_flat.std():.4f}')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at 0
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    if title:
        plt.suptitle(title, fontsize=14, y=1.00)
    else:
        plt.suptitle('Model Weight Distributions', fontsize=14, y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_feature_analysis(features: np.ndarray,
                               labels: np.ndarray,
                               attention_weights: Optional[np.ndarray] = None,
                               model_weights: Optional[Dict[str, np.ndarray]] = None,
                               save_dir: str = './figures',
                               methods: List[str] = ['pca', 'tsne'],
                               show: bool = False) -> Dict[str, Optional[plt.Figure]]:
    """
    Generate all feature analysis visualizations at once.
    
    Args:
        features: Shape (n_samples, n_features)
        labels: Shape (n_samples,)
        attention_weights: Optional attention weight matrices
        model_weights: Optional dictionary of model weights
        save_dir: Directory to save visualizations
        methods: List of dimensionality reduction methods to use
        show: Whether to display plots (default False for batch operations)
        
    Returns:
        Dictionary mapping visualization names to Figure objects
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    figures = {}
    
    print("\n" + "="*60)
    print("GENERATING FEATURE ANALYSIS VISUALIZATIONS")
    print("="*60)
    
    # 1. Embedding space visualizations
    for method in methods:
        for n_components in [2, 3]:
            vis_name = f'embedding_{method}_{n_components}d'
            save_path = os.path.join(save_dir, f'{vis_name}.png')
            
            try:
                fig = visualize_embedding_space(
                    features, labels, 
                    method=method, 
                    n_components=n_components,
                    save_path=save_path,
                    show=show
                )
                figures[vis_name] = fig
                if fig:
                    plt.close(fig)
            except Exception as e:
                warnings.warn(f"Failed to create {vis_name}: {e}")
    
    # 2. Attention maps
    if attention_weights is not None:
        vis_name = 'attention_maps'
        save_path = os.path.join(save_dir, f'{vis_name}.png')
        
        try:
            fig = visualize_attention_maps(
                attention_weights,
                save_path=save_path,
                show=show
            )
            figures[vis_name] = fig
            if fig:
                plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to create attention maps: {e}")
    
    # 3. Weight distributions
    if model_weights is not None:
        vis_name = 'weight_distributions'
        save_path = os.path.join(save_dir, f'{vis_name}.png')
        
        try:
            fig = visualize_weight_distributions(
                model_weights,
                save_path=save_path,
                show=show
            )
            figures[vis_name] = fig
            if fig:
                plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to create weight distributions: {e}")
    
    print("\n✓ Visualization generation complete!")
    print(f"  Saved {len([f for f in figures.values() if f is not None])} visualizations to {save_dir}")
    print("="*60)
    
    return figures
