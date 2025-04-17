import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import os
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F

def visualize_feature_space(model, data_loader, save_dir='feature_viz', n_samples=100):
    """
    Generate t-SNE and UMAP visualizations of the feature space
    
    Args:
        model: The Few-Shot Transformer model
        data_loader: DataLoader containing samples to visualize
        save_dir: Directory to save visualizations
        n_samples: Maximum number of samples to use (for faster computation)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Collect features and labels
    features = []
    labels = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if len(features) >= n_samples:
                break
                
            x = x.to(device)
            
            # Extract features before prototype aggregation
            z_support, z_query = model.parse_feature(x, is_feature=False)
            
            # Reshape to [n_samples, feature_dim]
            z_all = torch.cat([z_support, z_query], dim=0)
            z_all = z_all.view(-1, z_all.size(-1))
            
            # Get actual class labels (repeating support/query structure)
            batch_size = x.size(0)
            y_all = torch.arange(model.n_way).repeat_interleave(model.k_shot + model.n_query)
            
            # Add to collection
            features.append(z_all.cpu().numpy())
            labels.append(y_all.numpy())
    
    # Combine all batches
    features = np.vstack(features[:n_samples])
    labels = np.hstack(labels[:n_samples])
    
    # Generate t-SNE visualization
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                         c=labels, alpha=0.6, cmap='tab10')
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Feature Space')
    plt.savefig(f"{save_dir}/tsne_features.png")
    plt.close()
    
    # Generate UMAP visualization
    print("Computing UMAP embedding...")
    try:
        reducer = umap.UMAP(random_state=42)
        umap_result = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                             c=labels, alpha=0.6, cmap='tab10')
        plt.colorbar(scatter, label='Class')
        plt.title('UMAP Visualization of Feature Space')
        plt.savefig(f"{save_dir}/umap_features.png")
        plt.close()
    except ImportError:
        print("UMAP visualization skipped (install umap-learn package to enable)")

def visualize_prototypes(model, data_loader, save_dir='feature_viz'):
    """
    Visualize the distribution of prototype features and their relationships
    
    Args:
        model: The Few-Shot Transformer model
        data_loader: DataLoader containing episodes
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Collect prototypes from multiple episodes
    all_protos = []
    
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= 10:  # Limit to 10 episodes
                break
                
            x = x.to(device)
            
            # Extract support features and compute prototypes
            z_support, _ = model.parse_feature(x, is_feature=False)
            z_support = z_support.contiguous().view(model.n_way, model.k_shot, -1)
            z_proto = (z_support * model.sm(model.proto_weight)).sum(1)  # [n_way, feat_dim]
            
            all_protos.append(z_proto.cpu().numpy())
    
    all_protos = np.vstack(all_protos)  # [n_episodes * n_way, feat_dim]
    
    # 1. Feature activation strength for each prototype
    avg_proto = all_protos.mean(axis=0)
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(min(100, len(avg_proto))), np.sort(avg_proto)[-100:])
    plt.title('Top 100 Feature Activations (Averaged Across Prototypes)')
    plt.xlabel('Feature Index (sorted by strength)')
    plt.ylabel('Activation Strength')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/prototype_feature_strength.png")
    plt.close()
    
    # 2. Prototype similarity heatmap
    episode_protos = all_protos[:model.n_way]  # Take prototypes from first episode
    similarity = np.zeros((model.n_way, model.n_way))
    
    for i in range(model.n_way):
        for j in range(model.n_way):
            # Cosine similarity
            cos_sim = np.dot(episode_protos[i], episode_protos[j]) / (
                np.linalg.norm(episode_protos[i]) * np.linalg.norm(episode_protos[j]))
            similarity[i, j] = cos_sim
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Prototype Cosine Similarity')
    plt.xlabel('Class Index')
    plt.ylabel('Class Index')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/prototype_similarity.png")
    plt.close()

def visualize_attention_components(model, data_loader, save_dir='feature_viz'):
    """
    Visualize how different attention components (cosine, covariance, variance)
    contribute to the final attention scores
    
    Args:
        model: The Few-Shot Transformer model
        data_loader: DataLoader containing episodes
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get a single episode
    x, _ = next(iter(data_loader))
    x = x.to(device)
    
    # Store original forward method
    original_forward = model.ATTN.forward
    
    # Components to track
    cos_scores = None
    cov_scores = None
    var_scores = None
    component_weights = None
    
    def modified_forward(self, q, k, v):
        nonlocal cos_scores, cov_scores, var_scores, component_weights
        
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) ->  h q n d', h = self.heads), (q, k ,v))
        
        if self.variant == "cosine":
            # Calculate the components
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            
            q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
            k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
            cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
            cov_component = cov_component / f_q.size(-1)
            
            q_var = torch.var(f_q, dim=-1, keepdim=True)
            k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)
            var_scale = F.sigmoid(self.var_scale) * 3.0
            var_component = torch.matmul(q_var, k_var)
            var_component = var_component * var_scale / f_q.size(-1)
            
            if self.dynamic_weight:
                q_global = f_q.mean(dim=(1, 2))
                k_global = f_k.mean(dim=(1, 2))
                k_shot_feat = torch.full((self.heads, 1), float(self.k_shot) / 10.0, device=q_global.device)
                
                qk_features = torch.cat([q_global, k_global, k_shot_feat], dim=-1)
                weights = self.weight_predictor(qk_features)
                
                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
                
                # Store for visualization
                cos_scores = (cos_weight * cosine_sim).detach().cpu()
                cov_scores = (cov_weight * cov_component).detach().cpu()
                var_scores = (var_weight * var_component).detach().cpu()
                component_weights = weights.detach().cpu()
                
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
            else:
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = 1.0 - cov_weight - var_weight
                
                # Store for visualization
                cos_scores = (cos_weight * cosine_sim).detach().cpu()
                cov_scores = (cov_weight * cov_component).detach().cpu()
                var_scores = (var_weight * var_component).detach().cpu()
                component_weights = torch.tensor([[cos_weight, cov_weight, var_weight]])
                
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
                
            out = torch.matmul(dots, f_v)
        else:
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale            
            out = torch.matmul(self.sm(dots), f_v)
        
        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)
    
    # Replace forward method
    import types
    model.ATTN.forward = types.MethodType(modified_forward, model.ATTN)
    
    # Run forward pass
    with torch.no_grad():
        _ = model.set_forward(x)
    
    # Restore original forward method
    model.ATTN.forward = original_forward
    
    # Check if we got component data
    if cos_scores is None:
        print("Failed to capture attention component data")
        return
    
    # Visualize component weights
    plt.figure(figsize=(10, 6))
    components = ['Cosine', 'Covariance', 'Variance']
    
    weights = component_weights.numpy()
    if weights.shape[0] > 1:  # Multiple attention heads
        plt.bar(components, weights.mean(axis=0))
        plt.errorbar(components, weights.mean(axis=0), weights.std(axis=0), fmt='none', color='black', capsize=5)
    else:
        plt.bar(components, weights[0])
    
    plt.title('Attention Component Weights')
    plt.ylabel('Weight')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/component_weights.png")
    plt.close()
    
    # Visualize attention scores
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Average over heads for visualization
    cos_avg = cos_scores.mean(dim=0)[0, 0].numpy()
    cov_avg = cov_scores.mean(dim=0)[0, 0].numpy()
    var_avg = var_scores.mean(dim=0)[0, 0].numpy()
    
    im0 = axes[0].imshow(cos_avg, cmap='viridis')
    axes[0].set_title('Cosine Component')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(cov_avg, cmap='viridis')
    axes[1].set_title('Covariance Component')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(var_avg, cmap='viridis')
    axes[2].set_title('Variance Component')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/component_attention_scores.png")
    plt.close()

def visualize_feature_activations(model, img_loader, save_dir='feature_viz', n_imgs=5):
    """
    Visualize which parts of input images activate the model's features
    
    Args:
        model: The Few-Shot Transformer model with backbone
        img_loader: DataLoader containing images (should return original images)
        save_dir: Directory to save visualizations
        n_imgs: Number of images to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get a few images
    imgs = []
    for i, (x, _) in enumerate(img_loader):
        if i >= n_imgs:
            break
        imgs.append(x[0])  # Take the first image from each batch
    
    if not imgs:
        print("No images found for activation visualization")
        return
        
    # Register hooks to get feature maps
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Find a suitable layer in the backbone to hook
    # Typically the last convolutional layer before features are extracted
    # This may need to be adjusted based on your specific backbone
    if hasattr(model.feature, 'layer4'):
        # ResNet backbone
        hook = model.feature.layer4.register_forward_hook(get_activation('feature'))
    elif hasattr(model.feature, 'conv4'):
        # Conv4 backbone
        hook = model.feature.conv4.register_forward_hook(get_activation('feature'))
    else:
        print("Could not find a suitable layer to hook for activation visualization")
        return
    
    # Process each image
    for i, img in enumerate(imgs):
        img = img.unsqueeze(0).to(device)  # Add batch dimension
        
        # Forward pass
        with torch.no_grad():
            _ = model.feature_forward(img)
        
        # Get the feature maps
        feature_maps = activation['feature']
        
        # Sum across channels for visualization
        activation_map = feature_maps.sum(dim=1).squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
        
        # Original image (convert from normalized to display format)
        img_np = img.cpu().squeeze().numpy()
        if img_np.shape[0] == 3:  # If channels-first format
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Resize activation map to match image size
        from scipy.ndimage import zoom
        zoom_factors = (img_np.shape[0] / activation_map.shape[0], 
                        img_np.shape[1] / activation_map.shape[1])
        activation_map = zoom(activation_map, zoom_factors)
        
        # Create overlay
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(activation_map, cmap='hot')
        plt.title('Activation Map')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(img_np)
        plt.imshow(activation_map, cmap='hot', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/activation_map_{i}.png")
        plt.close()
    
    # Remove the hook
    hook.remove()

def visualize_shot_influence(model, data_loader, n_way=5, k_shot_values=[1, 5, 10], save_dir='feature_viz'):
    """
    Visualize how the number of shots influences the dynamic weights
    
    Args:
        model: The Few-Shot Transformer model
        data_loader: DataLoader containing episodes
        n_way: Number of ways
        k_shot_values: List of shot values to test
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    original_k_shot = model.k_shot
    
    # Storage for results
    shot_weights = {k: [] for k in k_shot_values}
    
    # Modify the model's forward function to use different k_shot values
    original_forward = model.ATTN.forward
    
    def modified_forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) ->  h q n d', h = self.heads), (q, k ,v))
        
        if self.variant == "cosine":
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            
            q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
            k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
            cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
            cov_component = cov_component / f_q.size(-1)
            
            q_var = torch.var(f_q, dim=-1, keepdim=True)
            k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)
            var_scale = F.sigmoid(self.var_scale) * 3.0
            var_component = torch.matmul(q_var, k_var)
            var_component = var_component * var_scale / f_q.size(-1)
            
            if self.dynamic_weight:
                q_global = f_q.mean(dim=(1, 2))
                k_global = f_k.mean(dim=(1, 2))
                k_shot_feat = torch.full((self.heads, 1), float(self.k_shot) / 10.0, device=q_global.device)
                
                qk_features = torch.cat([q_global, k_global, k_shot_feat], dim=-1)
                weights = self.weight_predictor(qk_features)
                
                # Store weights for visualization
                if not self.training:
                    shot_weights[self.k_shot].append(weights.detach().cpu().numpy().mean(axis=0))
                
                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)
                
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
            else:
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = 1.0 - cov_weight - var_weight
                
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
                
            out = torch.matmul(dots, f_v)
        else:
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale            
            out = torch.matmul(self.sm(dots), f_v)
        
        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)
    
    # Replace the forward method
    import types
    model.ATTN.forward = types.MethodType(modified_forward, model.ATTN)
    
    # Test with different k_shot values
    for k_shot in k_shot_values:
        model.ATTN.k_shot = k_shot
        model.k_shot = k_shot
        
        # Process a few episodes
        with torch.no_grad():
            model.eval()
            for i, (x, _) in enumerate(data_loader):
                if i >= 5:  # Limit to 5 episodes per k_shot value
                    break
                x = x.to(device)
                _ = model.set_forward(x)
    
    # Restore the original settings
    model.ATTN.forward = original_forward
    model.k_shot = original_k_shot
    model.ATTN.k_shot = original_k_shot
    
    # Calculate average weights for each k_shot value
    avg_weights = {}
    for k, weights_list in shot_weights.items():
        if weights_list:
            avg_weights[k] = np.mean(np.vstack(weights_list), axis=0)
    
    if not avg_weights:
        print("No weights were collected!")
        return
    
    # Visualize the results
    k_values = list(avg_weights.keys())
    components = ['Cosine', 'Covariance', 'Variance']
    
    plt.figure(figsize=(12, 6))
    width = 0.25
    
    # Plot bars for each component
    for i, comp in enumerate(components):
        values = [avg_weights[k][i] for k in k_values]
        plt.bar([x + i*width for x in range(len(k_values))], values, width, label=comp)
    
    plt.xlabel('Number of Shots (k)')
    plt.ylabel('Weight Value')
    plt.title('Effect of Shot Count on Attention Component Weights')
    plt.xticks([x + width for x in range(len(k_values))], k_values)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shot_influence.png")
    plt.close()

def run_all_visualizations(model, data_loader, img_loader=None, save_dir='feature_viz'):
    """
    Run all feature space visualizations
    
    Args:
        model: The Few-Shot Transformer model
        data_loader: DataLoader containing episodes
        img_loader: DataLoader containing original images (optional)
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("1. Visualizing feature space embeddings...")
    visualize_feature_space(model, data_loader, save_dir)
    
    print("2. Visualizing prototype relationships...")
    visualize_prototypes(model, data_loader, save_dir)
    
    print("3. Visualizing attention components...")
    visualize_attention_components(model, data_loader, save_dir)
    
    if img_loader is not None:
        print("4. Visualizing feature activations...")
        visualize_feature_activations(model, img_loader, save_dir)
    
    print("5. Visualizing influence of shot count...")
    visualize_shot_influence(model, data_loader, k_shot_values=[1, 5], save_dir=save_dir)
    
    print(f"All visualizations saved to {save_dir}/")