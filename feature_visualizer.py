import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
                batch_features = self._forward_hook_features(x)
                
                # Move to CPU and convert to numpy
                batch_features = batch_features.cpu().numpy()
                features.append(batch_features)
                
                # Generate labels from episodic structure
                # For few-shot learning with [n_way, n_shot+n_query] structure
                if len(x.shape) == 5:
                    n_way, n_samples = x.shape[:2]
                    # Create labels: each class has n_samples examples
                    episode_labels = np.repeat(np.arange(n_way), n_samples)
                    labels.append(episode_labels)
                else:
                    # Use provided labels
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
            x: Input tensor - expects [n_way, n_shot+n_query, channels, height, width]
            
        Returns:
            features: Extracted feature tensor
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
                 interactive=True, title=None, save_path=None, show=False, **kwargs):
        """
        Visualize the embeddings
        
        Args:
            embeddings: Pre-computed embeddings (if None, will compute using reduce_dimensions)
            labels: Labels for coloring (if None, will use self.labels)
            method: Method used for dimensionality reduction (for title)
            interactive: Whether to use plotly for interactive visualization
            title: Title for the plot
            save_path: Path to save the visualization
            show: Whether to display the plot using plt.show() (only for matplotlib)
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
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
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
                ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
                
            plt.title(title)
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Show if requested
            if show:
                plt.show()
                
            return plt.gcf()


    def visualize_all_projections(self, n_components_2d=2, n_components_3d=3, 
                                  show=True, save_dir=None, figsize=(18, 12)):
        """
        Generate comprehensive visualizations with PCA, t-SNE, and UMAP in both 2D and 3D.
        
        Args:
            n_components_2d: Number of components for 2D projection (should be 2)
            n_components_3d: Number of components for 3D projection (should be 3)
            show: Whether to display plots using plt.show()
            save_dir: Directory to save the plots (if None, plots are not saved)
            figsize: Figure size for the combined plot
            
        Returns:
            Dictionary containing all embeddings and the figure object
        """
        if self.features is None or len(self.features) == 0:
            raise ValueError("No features available. Call extract_features() first.")
        
        # Create comprehensive figure with subplots
        methods = ['PCA', 't-SNE', 'UMAP']
        projections = {}
        
        # Create 2x3 grid: 2 rows (2D, 3D) x 3 columns (PCA, t-SNE, UMAP)
        fig = plt.figure(figsize=figsize)
        
        for idx, method in enumerate(methods):
            method_lower = method.lower().replace('-', '')
            
            # 2D projection
            print(f"Computing {method} 2D projection...")
            embeddings_2d = self.reduce_dimensions(method=method_lower, n_components=n_components_2d)
            projections[f'{method}_2D'] = embeddings_2d
            
            ax = fig.add_subplot(2, 3, idx + 1)
            self._plot_2d_scatter(ax, embeddings_2d, self.labels, f'{method} 2D Projection')
            
            # 3D projection
            print(f"Computing {method} 3D projection...")
            embeddings_3d = self.reduce_dimensions(method=method_lower, n_components=n_components_3d)
            projections[f'{method}_3D'] = embeddings_3d
            
            ax = fig.add_subplot(2, 3, idx + 4, projection='3d')
            self._plot_3d_scatter(ax, embeddings_3d, self.labels, f'{method} 3D Projection')
        
        plt.tight_layout()
        
        # Save if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'feature_projections_all.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined visualization to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return {
            'projections': projections,
            'figure': fig
        }
    
    def _plot_2d_scatter(self, ax, embeddings, labels, title):
        """Helper function to plot 2D scatter plot."""
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                      color=colors[i], label=f'Class {label}', alpha=0.6, s=30)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.grid(True, alpha=0.3)
    
    def _plot_3d_scatter(self, ax, embeddings, labels, title):
        """Helper function to plot 3D scatter plot."""
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], embeddings[mask, 2],
                      color=colors[i], label=f'Class {label}', alpha=0.6, s=30)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')


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


def visualize_features_from_results(features, labels, show=True, save_dir=None, 
                                   figsize=(18, 12), title_prefix="Feature Space"):
    """
    Standalone function to visualize features with PCA, t-SNE, and UMAP projections.
    Can be called directly with pre-extracted features.
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,) containing class labels
        show: Whether to display plots using plt.show()
        save_dir: Directory to save the plots (if None, plots are not saved)
        figsize: Figure size for the combined plot
        title_prefix: Prefix for plot titles
        
    Returns:
        Dictionary containing all embeddings and the figure object
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    print(f"\n{'='*60}")
    print("Generating Feature Space Visualizations")
    print(f"{'='*60}")
    print(f"Feature shape: {features.shape}")
    print(f"Number of unique classes: {len(np.unique(labels))}")
    
    methods = {
        'PCA': PCA,
        't-SNE': TSNE,
        'UMAP': lambda n_components: umap.UMAP(n_components=n_components, random_state=42)
    }
    
    projections = {}
    
    # Create 2x3 grid: 2 rows (2D, 3D) x 3 columns (PCA, t-SNE, UMAP)
    fig = plt.figure(figsize=figsize)
    
    for idx, (method_name, method_class) in enumerate(methods.items()):
        # 2D projection
        print(f"\n{method_name} 2D projection...")
        if method_name == 'PCA':
            reducer_2d = method_class(n_components=2, random_state=42)
        elif method_name == 't-SNE':
            reducer_2d = method_class(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        else:  # UMAP
            reducer_2d = method_class(2)
        
        embeddings_2d = reducer_2d.fit_transform(features)
        projections[f'{method_name}_2D'] = embeddings_2d
        
        # Plot 2D
        ax = fig.add_subplot(2, 3, idx + 1)
        _plot_2d_scatter(ax, embeddings_2d, labels, f'{title_prefix} - {method_name} 2D')
        
        # 3D projection
        print(f"{method_name} 3D projection...")
        if method_name == 'PCA':
            reducer_3d = method_class(n_components=3, random_state=42)
        elif method_name == 't-SNE':
            reducer_3d = method_class(n_components=3, random_state=42, perplexity=min(30, len(features)-1))
        else:  # UMAP
            reducer_3d = method_class(3)
        
        embeddings_3d = reducer_3d.fit_transform(features)
        projections[f'{method_name}_3D'] = embeddings_3d
        
        # Plot 3D
        ax = fig.add_subplot(2, 3, idx + 4, projection='3d')
        _plot_3d_scatter(ax, embeddings_3d, labels, f'{title_prefix} - {method_name} 3D')
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'feature_projections_all.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved combined visualization to: {save_path}")
    
    # Show if requested
    if show:
        print(f"\n{'='*60}")
        print("Displaying visualizations...")
        print(f"{'='*60}\n")
        plt.show()
    
    return {
        'projections': projections,
        'figure': fig
    }


def _plot_2d_scatter(ax, embeddings, labels, title):
    """Helper function to plot 2D scatter plot."""
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                  color=colors[i], label=f'Class {label}', alpha=0.6, s=30, edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel('Component 1', fontsize=10)
    ax.set_ylabel('Component 2', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)


def _plot_3d_scatter(ax, embeddings, labels, title):
    """Helper function to plot 3D scatter plot."""
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], embeddings[mask, 2],
                  color=colors[i], label=f'Class {label}', alpha=0.6, s=30, edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel('Component 1', fontsize=9)
    ax.set_ylabel('Component 2', fontsize=9)
    ax.set_zlabel('Component 3', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)