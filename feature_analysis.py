"""
Feature Analysis Module for Few-Shot Learning

This module provides comprehensive feature space analysis including:
- Feature collapse detection
- Feature utilization metrics
- Diversity scores
- Feature redundancy analysis
- Intra-class consistency
- Confusing class pair identification
- Imbalance ratio calculation

Author: Extended evaluation implementation
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist, euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert (can be numpy array, scalar, dict, list, etc.)
    
    Returns:
        Object with all numpy types converted to native Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def compute_confidence_interval(accuracies: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute confidence interval for accuracy from multiple episodes.
    
    Args:
        accuracies: Array of accuracy values from test episodes
        confidence: Confidence level (default 0.95 for 95%)
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    mean = float(np.mean(accuracies))
    std = float(np.std(accuracies))
    n = len(accuracies)
    
    # Using normal approximation for large n
    z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    margin = float(z_score * std / np.sqrt(n))
    
    return mean, float(mean - margin), float(mean + margin)


def detect_feature_collapse(features: np.ndarray, threshold: float = 1e-4) -> Dict:
    """
    Detect feature dimensions that have collapsed (very low variance).
    
    Args:
        features: Feature matrix (n_samples, n_features)
        threshold: Variance threshold for collapse detection
    
    Returns:
        Dictionary with collapse statistics
    """
    std_per_dim = np.std(features, axis=0)
    collapsed_dims = np.sum(std_per_dim < threshold)
    total_dims = features.shape[1]
    
    result = {
        'collapsed_dimensions': int(collapsed_dims),
        'total_dimensions': int(total_dims),
        'collapse_ratio': float(collapsed_dims / total_dims),
        'std_per_dimension': std_per_dim.tolist(),
        'min_std': float(np.min(std_per_dim)),
        'max_std': float(np.max(std_per_dim)),
        'mean_std': float(np.mean(std_per_dim))
    }
    return convert_to_serializable(result)


def compute_feature_utilization(features: np.ndarray) -> Dict:
    """
    Compute feature utilization based on actual value distribution vs max range.
    
    Args:
        features: Feature matrix (n_samples, n_features)
    
    Returns:
        Dictionary with utilization metrics
    """
    # Compute range for each dimension
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    ranges = max_vals - min_vals
    
    # Compute utilization as the ratio of actual range to theoretical max
    # Using percentile-based approach to avoid outlier effects
    p5 = np.percentile(features, 5, axis=0)
    p95 = np.percentile(features, 95, axis=0)
    effective_range = p95 - p5
    
    # Utilization score per dimension
    utilization = np.where(ranges > 1e-8, effective_range / (ranges + 1e-8), 0)
    
    result = {
        'mean_utilization': float(np.mean(utilization)),
        'std_utilization': float(np.std(utilization)),
        'min_utilization': float(np.min(utilization)),
        'max_utilization': float(np.max(utilization)),
        'low_utilization_dims': int(np.sum(utilization < 0.3))
    }
    return convert_to_serializable(result)


def compute_diversity_score(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute diversity score as coefficient of variation of distances to class centroids.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Class labels for each sample
    
    Returns:
        Dictionary with diversity metrics
    """
    unique_labels = np.unique(labels)
    diversity_scores = []
    
    for label in unique_labels:
        class_features = features[labels == label]
        if len(class_features) < 2:
            continue
            
        # Compute centroid
        centroid = np.mean(class_features, axis=0, keepdims=True)
        
        # Compute distances to centroid
        distances = cdist(class_features, centroid, metric='euclidean').flatten()
        
        # Coefficient of variation (std / mean)
        if np.mean(distances) > 1e-8:
            cv = np.std(distances) / np.mean(distances)
            diversity_scores.append(cv)
    
    result = {
        'mean_diversity': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        'std_diversity': float(np.std(diversity_scores)) if diversity_scores else 0.0,
        'per_class_diversity': diversity_scores
    }
    return convert_to_serializable(result)


def analyze_feature_redundancy(features: np.ndarray, 
                               high_corr_threshold: float = 0.9,
                               moderate_corr_threshold: float = 0.7,
                               variance_threshold: float = 0.95) -> Dict:
    """
    Analyze feature redundancy through correlation and PCA.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        high_corr_threshold: Threshold for high correlation
        moderate_corr_threshold: Threshold for moderate correlation
        variance_threshold: Variance threshold for PCA (e.g., 0.95 for 95%)
    
    Returns:
        Dictionary with redundancy metrics
    """
    n_features = features.shape[1]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(features.T)
    
    # Find highly correlated pairs (excluding diagonal)
    high_corr_pairs = []
    moderate_corr_pairs = []
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr_val = abs(corr_matrix[i, j])
            if corr_val > high_corr_threshold:
                high_corr_pairs.append((i, j, corr_val))
            elif corr_val > moderate_corr_threshold:
                moderate_corr_pairs.append((i, j, corr_val))
    
    # PCA for effective dimensionality
    try:
        pca = PCA()
        pca.fit(features)
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        effective_dims = int(np.argmax(cumsum_variance >= variance_threshold) + 1)
    except:
        effective_dims = n_features
    
    result = {
        'total_features': int(n_features),
        'high_correlation_pairs': len(high_corr_pairs),
        'moderate_correlation_pairs': len(moderate_corr_pairs),
        'effective_dimensions_95pct': int(effective_dims),
        'dimensionality_reduction_ratio': float(effective_dims / n_features),
        'mean_abs_correlation': float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))
    }
    return convert_to_serializable(result)


def compute_intraclass_consistency(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute intra-class consistency using Euclidean distance and cosine similarity.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Class labels for each sample
    
    Returns:
        Dictionary with consistency metrics
    """
    unique_labels = np.unique(labels)
    euclidean_consistencies = []
    cosine_consistencies = []
    
    for label in unique_labels:
        class_features = features[labels == label]
        if len(class_features) < 2:
            continue
        
        # Euclidean distance-based consistency (lower distance = higher consistency)
        pairwise_distances = cdist(class_features, class_features, metric='euclidean')
        # Take upper triangle (excluding diagonal)
        upper_tri_indices = np.triu_indices_from(pairwise_distances, k=1)
        avg_distance = np.mean(pairwise_distances[upper_tri_indices])
        euclidean_consistencies.append(avg_distance)
        
        # Cosine similarity-based consistency (higher similarity = higher consistency)
        # Normalize features
        normalized = class_features / (np.linalg.norm(class_features, axis=1, keepdims=True) + 1e-8)
        cosine_similarities = np.dot(normalized, normalized.T)
        avg_cosine = np.mean(cosine_similarities[upper_tri_indices])
        cosine_consistencies.append(avg_cosine)
    
    # Convert to consistency scores (normalize and invert Euclidean)
    if euclidean_consistencies:
        max_dist = max(euclidean_consistencies)
        if max_dist > 0:
            euclidean_consistency_scores = [1 - (d / max_dist) for d in euclidean_consistencies]
        else:
            euclidean_consistency_scores = [1.0] * len(euclidean_consistencies)
    else:
        euclidean_consistency_scores = []
    
    # Combined consistency score (average of normalized Euclidean and cosine)
    combined_scores = []
    for euc, cos in zip(euclidean_consistency_scores, cosine_consistencies):
        combined_scores.append((euc + cos) / 2)
    
    result = {
        'mean_euclidean_consistency': float(np.mean(euclidean_consistency_scores)) if euclidean_consistency_scores else 0.0,
        'mean_cosine_consistency': float(np.mean(cosine_consistencies)) if cosine_consistencies else 0.0,
        'mean_combined_consistency': float(np.mean(combined_scores)) if combined_scores else 0.0,
        'per_class_euclidean': euclidean_consistencies,
        'per_class_cosine': cosine_consistencies,
        'per_class_combined': combined_scores
    }
    return convert_to_serializable(result)


def identify_confusing_pairs(features: np.ndarray, labels: np.ndarray, 
                            top_k: int = 5) -> Dict:
    """
    Identify confusing class pairs based on inter-centroid distance.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Class labels for each sample
        top_k: Number of top confusing pairs to return
    
    Returns:
        Dictionary with confusing pair information
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Compute centroids for each class
    centroids = []
    for label in unique_labels:
        class_features = features[labels == label]
        centroid = np.mean(class_features, axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Compute pairwise centroid distances
    centroid_distances = cdist(centroids, centroids, metric='euclidean')
    
    # Extract pairs with their distances (excluding diagonal)
    pairs = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            pairs.append({
                'class_1': int(unique_labels[i]),
                'class_2': int(unique_labels[j]),
                'distance': float(centroid_distances[i, j])
            })
    
    # Sort by distance (smaller = more confusing)
    pairs.sort(key=lambda x: x['distance'])
    
    result = {
        'most_confusing_pairs': pairs[:top_k],
        'mean_intercentroid_distance': float(np.mean([p['distance'] for p in pairs])),
        'std_intercentroid_distance': float(np.std([p['distance'] for p in pairs])),
        'min_intercentroid_distance': float(pairs[0]['distance']) if pairs else 0.0,
        'max_intercentroid_distance': float(pairs[-1]['distance']) if pairs else 0.0
    }
    return convert_to_serializable(result)


def compute_imbalance_ratio(labels: np.ndarray) -> Dict:
    """
    Compute imbalance ratio (minority class / majority class).
    
    Args:
        labels: Class labels for each sample
    
    Returns:
        Dictionary with imbalance metrics
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    min_count = np.min(counts)
    max_count = np.max(counts)
    
    imbalance_ratio = min_count / max_count if max_count > 0 else 1.0
    
    result = {
        'imbalance_ratio': float(imbalance_ratio),
        'min_class_samples': int(min_count),
        'max_class_samples': int(max_count),
        'mean_class_samples': float(np.mean(counts)),
        'std_class_samples': float(np.std(counts)),
        'per_class_counts': {int(label): int(count) for label, count in zip(unique_labels, counts)}
    }
    return convert_to_serializable(result)


def comprehensive_feature_analysis(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Perform comprehensive feature space analysis.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Class labels for each sample
    
    Returns:
        Dictionary containing all analysis results
    """
    results = {
        'feature_collapse': detect_feature_collapse(features),
        'feature_utilization': compute_feature_utilization(features),
        'diversity_score': compute_diversity_score(features, labels),
        'feature_redundancy': analyze_feature_redundancy(features),
        'intraclass_consistency': compute_intraclass_consistency(features, labels),
        'confusing_pairs': identify_confusing_pairs(features, labels),
        'imbalance_ratio': compute_imbalance_ratio(labels)
    }
    
    return convert_to_serializable(results)
