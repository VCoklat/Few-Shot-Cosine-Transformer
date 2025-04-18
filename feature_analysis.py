import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from einops import rearrange
from methods.transformer import cosine_distance
import torch.nn.functional as F

def setup_visualization_folder():
    """Create folders for visualizations"""
    os.makedirs("feature_analysis", exist_ok=True)
    os.makedirs("feature_analysis/1shot", exist_ok=True)
    os.makedirs("feature_analysis/5shot", exist_ok=True)

def analyze_feature_properties(model, data_loader, shot_setting, save_dir="feature_analysis"):
    """Analyze statistical properties of feature space that affect dynamic weights"""
    device = next(model.parameters()).device
    model.eval()
    
    # Containers for collected statistics
    cosine_similarities = []
    covariance_values = []
    variance_products = []
    feature_means = []
    feature_vars = []
    
    # Collect feature properties from several episodes
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= 10:  # Limit to 10 episodes
                break
                
            x = x.to(device)
            
            # Extract features from current episode
            z_support, z_query = model.parse_feature(x, is_feature=False)
            z_support = z_support.contiguous().view(model.n_way, model.k_shot, -1)
            
            # Create prototypes as in the model
            z_proto = (z_support * model.sm(model.proto_weight)).sum(1).unsqueeze(0)
            z_query = z_query.contiguous().view(model.n_way * model.n_query, -1).unsqueeze(1)
            
            # Process features through input linear layer as in Attention
            f_q, f_k = map(lambda t: rearrange(
                model.ATTN.input_linear(t), 'q n (h d) -> h q n d', h=model.ATTN.heads
            ), (z_proto, z_query))
            
            # Calculate the same statistics used by the dynamic weight predictor
            
            # 1. Cosine similarities
            cos_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            cosine_similarities.append(cos_sim.cpu().numpy())
            
            # 2. Covariance component 
            q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
            k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
            cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
            cov_component = cov_component / f_q.size(-1)
            covariance_values.append(cov_component.cpu().numpy())
            
            # 3. Variance component
            q_var = torch.var(f_q, dim=-1, keepdim=True)
            k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)
            var_scale = F.sigmoid(model.ATTN.var_scale) * 3.0
            var_component = torch.matmul(q_var, k_var) 
            var_component = var_component * var_scale / f_q.size(-1)
            variance_products.append(var_component.cpu().numpy())
            
            # 4. General feature statistics
            feature_means.append(f_q.mean().item())
            feature_vars.append(f_q.var().item())
    
    # Convert lists to arrays for analysis
    cosine_similarities = np.concatenate(cosine_similarities, axis=0)
    covariance_values = np.concatenate(covariance_values, axis=0)
    variance_products = np.concatenate(variance_products, axis=0)
    
    # Create visualizations
    plot_component_distributions(
        cosine_similarities, covariance_values, variance_products,
        f"{save_dir}/{shot_setting}shot/component_distributions.png"
    )
    
    plot_component_correlations(
        cosine_similarities, covariance_values, variance_products, 
        f"{save_dir}/{shot_setting}shot/component_correlations.png"
    )
    
    plot_feature_statistics(
        feature_means, feature_vars, f"{save_dir}/{shot_setting}shot/feature_statistics.png"
    )
    
    # Calculate and return key statistics
    return {
        "cosine_mean": float(np.mean(cosine_similarities)),
        "cosine_std": float(np.std(cosine_similarities)),
        "cov_mean": float(np.mean(covariance_values)),
        "cov_std": float(np.std(covariance_values)),
        "var_mean": float(np.mean(variance_products)),
        "var_std": float(np.std(variance_products)),
        "feature_mean": np.mean(feature_means),
        "feature_var": np.mean(feature_vars)
    }

def plot_component_distributions(cosine_similarities, covariance_values, variance_products, save_path):
    """Plot distributions of the three attention components"""
    # Flatten arrays for histogram
    cos_flat = cosine_similarities.flatten()
    cov_flat = covariance_values.flatten()
    var_flat = variance_products.flatten()
    
    # Create density plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(cos_flat, kde=True)
    plt.title(f'Cosine Similarity\n(mean={cos_flat.mean():.4f}, std={cos_flat.std():.4f})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    plt.subplot(1, 3, 2)
    sns.histplot(cov_flat, kde=True)
    plt.title(f'Covariance Component\n(mean={cov_flat.mean():.4f}, std={cov_flat.std():.4f})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    plt.subplot(1, 3, 3)
    sns.histplot(var_flat, kde=True)
    plt.title(f'Variance Component\n(mean={var_flat.mean():.4f}, std={var_flat.std():.4f})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_component_correlations(cosine_similarities, covariance_values, variance_products, save_path):
    """Plot correlations between the three attention components"""
    # Flatten arrays for scatter plots
    sample_size = min(5000, cosine_similarities.size)  # Limit points for clearer plots
    indices = np.random.choice(cosine_similarities.size, sample_size, replace=False)
    
    cos_flat = cosine_similarities.flatten()[indices]
    cov_flat = covariance_values.flatten()[indices]
    var_flat = variance_products.flatten()[indices]
    
    # Create correlation plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(cos_flat, cov_flat, alpha=0.5, s=5)
    plt.title(f'Cosine vs Covariance\ncorr={np.corrcoef(cos_flat, cov_flat)[0, 1]:.4f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Covariance')
    
    plt.subplot(1, 3, 2)
    plt.scatter(cos_flat, var_flat, alpha=0.5, s=5)
    plt.title(f'Cosine vs Variance\ncorr={np.corrcoef(cos_flat, var_flat)[0, 1]:.4f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Variance')
    
    plt.subplot(1, 3, 3)
    plt.scatter(cov_flat, var_flat, alpha=0.5, s=5) 
    plt.title(f'Covariance vs Variance\ncorr={np.corrcoef(cov_flat, var_flat)[0, 1]:.4f}')
    plt.xlabel('Covariance')
    plt.ylabel('Variance')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_statistics(feature_means, feature_vars, save_path):
    """Plot feature statistics across episodes"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(feature_means, bins=20)
    plt.title(f'Feature Means\n(avg={np.mean(feature_means):.4f}, std={np.std(feature_means):.4f})')
    plt.xlabel('Mean Value')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(feature_vars, bins=20)
    plt.title(f'Feature Variances\n(avg={np.mean(feature_vars):.4f}, std={np.std(feature_vars):.4f})')
    plt.xlabel('Variance Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_feature_clusters(model, data_loader, shot_setting, save_dir="feature_analysis"):
    """Visualize feature clusters for support and query samples"""
    device = next(model.parameters()).device
    model.eval()
    
    # Containers for collected features
    support_features = []
    proto_features = []
    query_features = []
    class_labels = []
    
    # Collect features from episodes
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= 10:  # Limit to 10 episodes for faster visualization
                break
                
            x = x.to(device)
            
            # Extract features from current episode
            z_support, z_query = model.parse_feature(x, is_feature=False)
            
            # Store original support features
            z_support_reshape = z_support.contiguous().view(model.n_way, model.k_shot, -1)
            support_features.append(z_support_reshape.cpu().numpy())
            
            # Create and store prototypes
            z_proto = (z_support_reshape * model.sm(model.proto_weight)).sum(1)
            proto_features.append(z_proto.cpu().numpy())
            
            # Store query features
            z_query_reshape = z_query.contiguous().view(model.n_way * model.n_query, -1)
            query_features.append(z_query_reshape.cpu().numpy())
            
            # Generate class labels
            labels = np.repeat(np.arange(model.n_way), model.n_query)
            class_labels.append(labels)
    
    # Combine across episodes
    support_features = np.vstack([f.reshape(-1, f.shape[-1]) for f in support_features])
    proto_features = np.vstack(proto_features)
    query_features = np.vstack(query_features)
    class_labels = np.hstack(class_labels)
    
    # Visualize with t-SNE
    visualize_tsne(
        support_features, proto_features, query_features, 
        f"{save_dir}/{shot_setting}shot/feature_clusters.png"
    )

def visualize_tsne(support_features, proto_features, query_features, save_path):
    """Create t-SNE visualization of feature space"""
    # Combine features for t-SNE (but keep track of which is which)
    combined_features = np.vstack([support_features, proto_features, query_features])
    
    # Create category labels
    support_markers = np.zeros(len(support_features))
    proto_markers = np.ones(len(proto_features))
    query_markers = np.ones(len(query_features)) * 2
    category_markers = np.concatenate([support_markers, proto_markers, query_markers])
    
    # Calculate t-SNE embedding
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(combined_features)
    
    # Split results back
    support_tsne = tsne_result[:len(support_features)]
    proto_tsne = tsne_result[len(support_features):len(support_features)+len(proto_features)]
    query_tsne = tsne_result[len(support_features)+len(proto_features):]
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot support samples
    plt.scatter(support_tsne[:, 0], support_tsne[:, 1], 
               c='blue', marker='o', alpha=0.5, label='Support')
    
    # Plot prototypes
    plt.scatter(proto_tsne[:, 0], proto_tsne[:, 1], 
               c='red', marker='*', s=200, label='Prototypes')
    
    # Plot query samples
    plt.scatter(query_tsne[:, 0], query_tsne[:, 1], 
               c='green', marker='x', alpha=0.7, label='Query')
    
    plt.title('t-SNE Feature Space Visualization')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_shot_settings(stats_1shot, stats_5shot, save_dir="feature_analysis"):
    """Compare key statistics between 1-shot and 5-shot settings"""
    # Create comparison bar chart
    plt.figure(figsize=(15, 8))
    
    # Define metrics to compare
    metrics = [
        ('cosine_mean', 'Cosine Mean'),
        ('cosine_std', 'Cosine Std'),
        ('cov_mean', 'Covariance Mean'),
        ('cov_std', 'Covariance Std'),
        ('var_mean', 'Variance Mean'),
        ('var_std', 'Variance Std'),
        ('feature_mean', 'Feature Mean'),
        ('feature_var', 'Feature Variance')
    ]
    
    # Extract values
    metric_keys = [m[0] for m in metrics]
    metric_names = [m[1] for m in metrics]
    values_1shot = [stats_1shot[k] for k in metric_keys]
    values_5shot = [stats_5shot[k] for k in metric_keys]
    
    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, values_1shot, width, label='1-shot')
    plt.bar(x + width/2, values_5shot, width, label='5-shot')
    
    # Add labels and legend
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Comparison of Feature Statistics Between 1-shot and 5-shot')
    plt.xticks(x, metric_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shot_comparison.png")
    plt.close()
    
    # Also create a ratio plot to visualize relative differences
    plt.figure(figsize=(10, 6))
    
    # Calculate ratios (5-shot / 1-shot)
    ratios = [s5/s1 if s1 != 0 else 0 for s5, s1 in zip(values_5shot, values_1shot)]
    
    # Plot ratios
    plt.bar(x, ratios)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Metric')
    plt.ylabel('Ratio (5-shot / 1-shot)')
    plt.title('Ratio of 5-shot to 1-shot Feature Statistics')
    plt.xticks(x, metric_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shot_ratio.png")
    plt.close()