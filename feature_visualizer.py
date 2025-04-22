import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from einops import rearrange

# Try importing UMAP, but make it optional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with 'pip install umap-learn' for additional visualization options.")

class FeatureVisualizer:
    def __init__(self, model, device='cuda'):
        """
        Initialize the feature visualizer
        
        Args:
            model: The trained model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.features = []
        self.labels = []
        self.feature_types = []  # To track if features are from support, query, or prototypes
        
    def extract_features(self, data_loader, layer='penultimate'):
        """
        Extract features from the model
        
        Args:
            data_loader: DataLoader containing samples
            layer: Which layer to extract features from
        
        Returns:
            features: List of feature vectors
            labels: List of corresponding labels
        """
        all_features = []
        all_labels = []
        all_feature_types = []
        
        # Reset containers
        self.features = []
        self.labels = []
        self.feature_types = []
        
        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm(data_loader, desc="Extracting features")):
                if i >= 10:  # Limit to 10 episodes for visualization clarity
                    break
                
                x = x.to(self.device)
                
                # Use model's parse_feature method if available (common in few-shot models)
                if hasattr(self.model, 'parse_feature'):
                    z_support, z_query = self.model.parse_feature(x, is_feature=False)
                    
                    # Reshape support features
                    z_support_reshape = z_support.contiguous().view(
                        self.model.n_way, self.model.n_support, -1)
                    
                    # Create and store prototypes if the model can create them
                    if hasattr(self.model, 'get_prototypes'):
                        z_proto = self.model.get_prototypes(z_support_reshape)
                    elif hasattr(self.model, 'proto_weight'):
                        # Specific to models with proto_weight attribute
                        z_proto = (z_support_reshape * self.model.sm(self.model.proto_weight)).sum(1)
                    else:
                        # Default: mean of support features
                        z_proto = z_support_reshape.mean(1)
                    
                    # Store query features
                    z_query_reshape = z_query.contiguous().view(
                        self.model.n_way * self.model.n_query, -1)
                    
                    # Convert tensors to numpy
                    support_np = z_support_reshape.view(-1, z_support_reshape.shape[-1]).cpu().numpy()
                    proto_np = z_proto.cpu().numpy()
                    query_np = z_query_reshape.cpu().numpy()
                    
                    # Generate labels
                    support_labels = np.repeat(np.arange(self.model.n_way), self.model.n_support)
                    proto_labels = np.arange(self.model.n_way)
                    query_labels = np.repeat(np.arange(self.model.n_way), self.model.n_query)
                    
                    # Combine all features
                    all_features.extend([support_np, proto_np, query_np])
                    all_labels.extend([support_labels, proto_labels, query_labels])
                    
                    # Mark feature types: 0=support, 1=prototype, 2=query
                    all_feature_types.extend([
                        np.zeros(len(support_np)),
                        np.ones(len(proto_np)),
                        np.ones(len(query_np)) * 2
                    ])
                else:
                    # Fallback for models without parse_feature
                    batch_features = self._forward_hook_features(x)
                    batch_features_np = batch_features.cpu().numpy()
                    
                    # For episodic format [n_way, n_shot+n_query, ...]
                    if len(x.shape) == 5:  
                        n_way, n_samples = x.shape[:2]
                        episode_labels = np.repeat(np.arange(n_way), n_samples)
                        
                        # All features are treated as queries in this case
                        feature_types = np.ones(len(batch_features_np)) * 2
                    else:
                        # Use sequential labels for non-episodic format
                        episode_labels = np.arange(len(batch_features_np))
                        feature_types = np.ones(len(batch_features_np)) * 2
                    
                    all_features.append(batch_features_np)
                    all_labels.append(episode_labels)
                    all_feature_types.append(feature_types)
        
        # Combine all episodes
        if any(isinstance(f, list) for f in all_features):
            # Handle the case where we have separate support/proto/query features
            self.features = np.vstack([f for sublist in all_features for f in sublist])
            self.labels = np.concatenate([l for sublist in all_labels for l in sublist])
            self.feature_types = np.concatenate([t for sublist in all_feature_types for t in sublist])
        else:
            # Handle the case where we have simple feature lists
            self.features = np.vstack(all_features)
            self.labels = np.concatenate(all_labels)
            self.feature_types = np.concatenate(all_feature_types)
        
        return self.features, self.labels, self.feature_types
    
    def _forward_hook_features(self, x):
        """
        Use forward hooks to extract features
        
        Args:
            x: Input tensor            
        Returns:
            features: Feature tensor
        """
        features = None
        
        # Reshape episode data to standard batch format
        if len(x.shape) == 5:  # [n_way, n_shot+n_query, channels, height, width]
            n_way, n_samples, channels, height, width = x.shape
            x_reshaped = x.view(-1, channels, height, width)  # Flatten to [n_way*n_samples, channels, height, width]
        else:
            x_reshaped = x
        
        def hook_fn(module, input, output):
            nonlocal features
            if isinstance(output, tuple):
                output = output[0]  # Some models return tuples
            features = output
            
            # Flatten if needed, but maintain batch dimension
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
        
        # Register hook to the appropriate layer based on model type
        if hasattr(self.model, 'feature_extractor'):
            handle = self.model.feature_extractor.register_forward_hook(hook_fn)
        elif hasattr(self.model, 'feature'):
            handle = self.model.feature.register_forward_hook(hook_fn)
        elif hasattr(self.model, 'backbone'):
            handle = self.model.backbone.register_forward_hook(hook_fn)
        else:
            raise AttributeError(
                f"Model {type(self.model).__name__} doesn't have a recognizable feature extraction component."
            )
        
        # Use the feature extractor directly instead of the full model
        try:
            if hasattr(self.model, 'feature'):
                _ = self.model.feature(x_reshaped)
            elif hasattr(self.model, 'feature_extractor'):
                _ = self.model.feature_extractor(x_reshaped)
            else:
                # Only use backbone as fallback
                _ = self.model.backbone(x_reshaped)
        except Exception as e:
            handle.remove()  # Clean up hook before re-raising
            raise RuntimeError(f"Error during feature extraction: {e}")
        
        # Remove the hook
        handle.remove()
        
        return features
        
    def reduce_dimensions(self, method='tsne', n_components=2, **kwargs):
        """
        Reduce dimensions for visualization
        
        Args:
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            n_components: Number of dimensions to reduce to (2 or 3)
            **kwargs: Additional arguments for the dimensionality reduction method
            
        Returns:
            embeddings: Reduced dimensionality embeddings
        """
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, **kwargs)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                print("UMAP not available. Falling back to t-SNE.")
                method = 'tsne'
                reducer = TSNE(n_components=n_components, random_state=42, **kwargs)
            else:
                reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
        embeddings = reducer.fit_transform(self.features)
        return embeddings
        
    def visualize(self, embeddings=None, labels=None, feature_types=None,
                 method='tsne', interactive=True, title=None, save_path=None, **kwargs):
        """
        Visualize the embeddings
        
        Args:
            embeddings: Pre-computed embeddings (if None, will compute using reduce_dimensions)
            labels: Labels for coloring (if None, will use self.labels)
            feature_types: Feature types for marking (0=support, 1=prototype, 2=query)
            method: Method used for dimensionality reduction (for title)
            interactive: Whether to use plotly for interactive visualization
            title: Title for the plot
            save_path: Path to save the visualization
            **kwargs: Additional arguments for plotting
        """
        if embeddings is None:
            embeddings = self.reduce_dimensions(method=method, **kwargs)
            
        if labels is None:
            labels = self.labels
            
        if feature_types is None:
            feature_types = self.feature_types
            
        if title is None:
            title = f"Feature Space Visualization ({method.upper()})"
            
        # Create a DataFrame for easy plotting
        df = pd.DataFrame()
        df['x'] = embeddings[:, 0]
        df['y'] = embeddings[:, 1]
        if embeddings.shape[1] > 2:
            df['z'] = embeddings[:, 2]
            
        df['label'] = labels
        df['feature_type'] = feature_types
        
        # Convert to string for better plotting
        df['label'] = df['label'].astype(str)
        
        # Set up color map for feature types
        feature_type_names = ['Support', 'Prototype', 'Query']
        
        # Make directories if needed
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        if interactive:
            # Use plotly for interactive visualization
            fig = px.scatter(
                df, x='x', y='y', 
                color='label', 
                symbol='feature_type',
                symbol_map={0: 'circle', 1: 'star', 2: 'x'},
                color_discrete_sequence=px.colors.qualitative.Set1,
                title=title, 
                hover_data=['label', 'feature_type']
            )
            
            # Update marker sizes (make prototypes larger)
            for i, ft in enumerate(df['feature_type'].unique()):
                symbol = 'star' if ft == 1 else ('circle' if ft == 0 else 'x')
                size = 15 if ft == 1 else 8
                fig.update_traces(
                    selector=dict(marker_symbol=symbol),
                    marker=dict(size=size),
                    name=feature_type_names[int(ft)]
                )
            
            # Update layout for better appearance
            fig.update_layout(
                legend_title="Class",
                template="plotly_white",
                margin=dict(l=20, r=20, b=20, t=40),
            )
            
            # Save if requested
            if save_path:
                fig.write_html(save_path)
                print(f"Interactive visualization saved to {save_path}")
                
            return fig
        else:
            # Use matplotlib for static visualization
            plt.figure(figsize=(12, 10))
            
            # Define markers and colors for feature types
            markers = ['o', '*', 'x']
            sizes = [30, 200, 50]
            
            # Get unique labels
            unique_labels = sorted(df['label'].unique())
            
            # Plot each feature type separately
            for ft, marker, size in zip(range(3), markers, sizes):
                ft_df = df[df['feature_type'] == ft]
                if len(ft_df) == 0:
                    continue
                    
                for i, label in enumerate(unique_labels):
                    mask = ft_df['label'] == label
                    if not mask.any():
                        continue
                        
                    plt.scatter(
                        ft_df.loc[mask, 'x'], 
                        ft_df.loc[mask, 'y'],
                        marker=marker,
                        s=size,
                        alpha=0.7,
                        label=f"{feature_type_names[ft]} (Class {label})" if ft == 0 else None
                    )
            
            plt.title(title, fontsize=16)
            plt.xlabel('Component 1', fontsize=12)
            plt.ylabel('Component 2', fontsize=12)
            
            # Create legend only for classes (support samples)
            h, l = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(l, h))
            plt.legend(by_label.values(), by_label.keys(), fontsize=10)
            
            # Add a separate legend for feature types
            plt.figtext(0.15, 0.03, "○ Support", fontsize=12)
            plt.figtext(0.35, 0.03, "★ Prototype", fontsize=12)
            plt.figtext(0.55, 0.03, "× Query", fontsize=12)
            
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Static visualization saved to {save_path}")
                
            return plt.gcf()
```

## How to Use This Code

Add the following function to your [`train_test.py`](train_test.py) file:

````python
def visualize_feature_space(model, params, test_loader=None):
    """Visualize the feature space of the model"""
    print("===================================")
    print("Feature Space Visualization: ")
    
    from feature_visualizer import FeatureVisualizer
    
    # Use the same test loader or create a new one if not provided
    if test_loader is None or len(test_loader) == 0:
        # Create a new test loader with a small number of episodes
        split = params.split
        if params.dataset == 'cross':
            if split == 'base':
                testfile = configs.data_dir['miniImagenet'] + 'all.json'
            else:
                testfile = configs.data_dir['CUB'] + split + '.json'
        elif params.dataset == 'cross_char':
            if split == 'base':
                testfile = configs.data_dir['Omniglot'] + 'noLatin.json'
            else:
                testfile = configs.data_dir['