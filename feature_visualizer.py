import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import torch
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
        
    def extract_features(self, data_loader, layer='penultimate'):
        """
        Extract features from the model
        
        Args:
            data_loader: DataLoader containing samples
            layer: Which layer to extract features from ('penultimate', 'encoder', etc.)
        
        Returns:
            features: List of feature vectors
            labels: List of corresponding labels
        """
        features = []
        labels = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(data_loader, desc="Extracting features")):
                x = x.to(self.device)
                
                # Model-specific feature extraction
                if hasattr(self.model, 'extract_features'):
                    # Use the model's built-in feature extraction method
                    batch_features = self.model.extract_features(x, layer=layer)
                else:
                    # For models without a specific feature extraction method,
                    # we can add a hook to capture intermediate activations
                    batch_features = self._forward_hook_features(x)
                    
                # Move to CPU and convert to numpy
                batch_features = batch_features.cpu().numpy()
                features.append(batch_features)
                
                # Store true labels (depends on your data format)
                if isinstance(y, torch.Tensor):
                    labels.append(y.cpu().numpy())
                else:
                    labels.append(y)
        
        # Concatenate all batches
        self.features = np.vstack(features) if len(features) > 1 else features[0]
        self.labels = np.concatenate(labels) if len(labels) > 1 else labels[0]
        
        return self.features, self.labels
    
    def _forward_hook_features(self, x):
        """
        Use forward hooks to extract features
        
        Args:
            x: Input tensor
            
        Returns:
            features: Feature tensor
        """
        # Example for models where we need to capture intermediate features 
        # using hooks (implement based on your model architecture)
        features = None
        
        def hook_fn(module, input, output):
            nonlocal features
            features = output.flatten(1)
            
        # Register hook to the appropriate layer
        if hasattr(self.model, 'feature_extractor'):
            # For FSCT or CTX, hook the feature extractor output
            handle = self.model.feature_extractor.register_forward_hook(hook_fn)
        else:
            # Generic fallback - assumes model.backbone is the feature extractor
            handle = self.model.backbone.register_forward_hook(hook_fn)
            
        # Forward pass to trigger the hook
        _ = self.model(x)
        
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
            reducer = TSNE(n_components=n_components, **kwargs)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
        embeddings = reducer.fit_transform(self.features)
        return embeddings
        
    def visualize(self, embeddings=None, labels=None, method='tsne', 
                 interactive=True, title=None, save_path=None, **kwargs):
        """
        Visualize the embeddings
        
        Args:
            embeddings: Pre-computed embeddings (if None, will compute using reduce_dimensions)
            labels: Labels for coloring (if None, will use self.labels)
            method: Method used for dimensionality reduction (for title)
            interactive: Whether to use plotly for interactive visualization
            title: Title for the plot
            save_path: Path to save the visualization
            **kwargs: Additional arguments for plotting
            
        Returns:
            fig: The figure object
        """
        if embeddings is None:
            embeddings = self.reduce_dimensions(method=method, **kwargs)
            
        if labels is None:
            labels = self.labels
            
        if title is None:
            title = f"Feature Space Visualization ({method.upper()})"
            
        # Create a DataFrame for easy plotting
        df = pd.DataFrame()
        df['x'] = embeddings[:, 0]
        df['y'] = embeddings[:, 1]
        if embeddings.shape[1] > 2:
            df['z'] = embeddings[:, 2]
            
        df['label'] = labels
        
        # Convert label to string for better formatting
        df['label'] = df['label'].astype(str)
            
        if interactive:
            # Use plotly for interactive visualization
            if embeddings.shape[1] == 2:
                fig = px.scatter(df, x='x', y='y', color='label', 
                                 title=title, hover_data=['label'])
            else:  # 3D
                fig = px.scatter_3d(df, x='x', y='y', z='z', color='label',
                                   title=title, hover_data=['label'])
                
            # Update layout for better appearance
            fig.update_layout(
                legend_title="Class",
                template="plotly_white",
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            # Save if requested
            if save_path:
                fig.write_html(save_path)
                
            return fig
        else:
            # Use matplotlib for static visualization
            plt.figure(figsize=(10, 8))
            
            if embeddings.shape[1] == 2:
                # 2D plot
                sns.scatterplot(data=df, x='x', y='y', hue='label', palette='viridis', **kwargs)
            else:
                # 3D plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Get unique labels and assign colors
                unique_labels = df['label'].unique()
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = df['label'] == label
                    ax.scatter(
                        df.loc[mask, 'x'], 
                        df.loc[mask, 'y'], 
                        df.loc[mask, 'z'],
                        color=colors[i],
                        label=label
                    )
                    
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_zlabel('Component 3')
                
            plt.title(title)
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return plt.gcf()
```

## 2. Add Model Extensions for Feature Extraction

You need to modify your model classes to support feature extraction. Add these methods to your model classes:

````python
# filepath: /workspaces/Few-Shot-Cosine-Transformer/methods/transformer.py
# Add this method to the FewShotTransformer class

def extract_features(self, x, layer='penultimate'):
    """
    Extract features from different layers of the model
    
    Args:
        x: Input tensor
        layer: Which layer to extract from ('embedding', 'penultimate', 'final')
        
    Returns:
        features: Extracted feature tensor
    """
    self.eval()
    with torch.no_grad():
        # Get feature embedding
        feature_maps = self.feature.forward(x)
        
        if layer == 'embedding':
            return feature_maps
            
        # Reshape if needed
        if self.feature.flatten:
            feature_size = feature_maps.size(1)
            # Get episode size (n_way * (n_support + n_query))
            n_support = self.n_support
            n_way = self.n_way
            n_query = x.size(0) - n_way*n_support
            
            if n_query < 1:  # If only support samples
                support_features = feature_maps
                return support_features
            
            support_features = feature_maps[:n_way*n_support].view(n_way, n_support, feature_size)
            query_features = feature_maps[n_way*n_support:].view(n_query, feature_size)
            
            if layer == 'penultimate':
                # For queries, just return the raw feature vectors
                return query_features
            
            # Process through transformer
            proto_features = self.encode_support_features(support_features)
            
            if layer == 'prototypes':
                return proto_features
                
            # Final layer output
            logits = self.get_classifier(proto_features, query_features)
            return logits
            
        else:
            # Handle case where flatten=False
            # Implementation depends on specific model architecture
            pass