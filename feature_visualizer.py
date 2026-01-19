# ==========================================
# Feature Visualizer (Visualisasi Fitur)
# ==========================================
# File ini menyediakan alat untuk memvisualisasikan representasi fitur yang dipelajari oleh model.
# Termasuk ekstraksi fitur, reduksi dimensi (t-SNE, PCA, UMAP), dan plot interaktif.

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
import cv2
import pandas as pd

class FeatureVisualizer:
    # Kelas utama untuk menangani visualisasi fitur model
    def __init__(self, model, device='cuda'):
        """
        Initialize the feature visualizer
        (Inisialisasi visualizer fitur)
        
        Args:
            model: The trained model (Model yang sudah dilatih)
            device: Device to run inference on (Perangkat untuk menjalankan inferensi: cpu/cuda)
        """
        self.model = model
        self.device = device
        self.model.eval() # Set model ke mode evaluasi
        self.features = [] # Menyimpan vektor fitur
        self.labels = []   # Menyimpan label
        # Hook storage (Penyimpanan untuk hook)
        self.gradients = None
        self.activations = None
        
    def extract_features(self, data_loader, layer='penultimate'):
        """
        Extract features from the model
        (Ekstraksi fitur dari model)
        
        Args:
            data_loader: DataLoader containing samples
            layer: Which layer to extract features from ('penultimate', 'encoder', etc.)
        
        Returns:
            features: List of feature vectors (Daftar vektor fitur)
            labels: List of corresponding labels (Daftar label yang sesuai)
        """
        features = []
        labels = []
        
        # Nonaktifkan perhitungan gradien untuk menghemat memori
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(data_loader, desc="Extracting features (Mengekstraksi fitur)")):
                x = x.to(self.device)
                
                # Model-specific feature extraction (Ekstraksi fitur spesifik model menggunakan hook)
                batch_features = self._forward_hook_features(x)
                
                # Move to CPU and convert to numpy (Pindahkan ke CPU dan konversi ke numpy)
                batch_features = batch_features.cpu().numpy()
                features.append(batch_features)
                
                # Generate labels from episodic structure (Generate label dari struktur episodik)
                # For few-shot learning with [n_way, n_shot+n_query] structure
                if len(x.shape) == 5:
                    n_way, n_samples = x.shape[:2]
                    # Create labels: each class has n_samples examples
                    # (Buat label: setiap kelas memiliki n_samples contoh)
                    episode_labels = np.repeat(np.arange(n_way), n_samples)
                    labels.append(episode_labels)
                else:
                    # Use provided labels (Gunakan label yang disediakan jika bukan struktur episodik)
                    if isinstance(y, torch.Tensor):
                        labels.append(y.cpu().numpy())
                    else:
                        labels.append(y)
        
        # Concatenate all batches (Gabungkan semua batch menjadi satu array)
        self.features = np.vstack(features) if len(features) > 1 else features[0]
        self.labels = np.concatenate(labels) if len(labels) > 1 else labels[0]
        
        return self.features, self.labels
    
    def _forward_hook_features(self, x):
        """
        Use forward hooks to extract features
        (Menggunakan forward hooks untuk mengambil output dari layer tertentu)
        
        Args:
            x: Input tensor - expects [n_way, n_shot+n_query, channels, height, width]
            
        Returns:
            features: Extracted feature tensor (Tensor fitur yang diekstraksi)
        """
        features = None
        
        # Reshape episode data to standard batch format (Ubah bentuk data episode ke format batch standar)
        if len(x.shape) == 5:  # [n_way, n_shot+n_query, channels, height, width]
            n_way, n_samples, channels, height, width = x.shape
            x_reshaped = x.view(-1, channels, height, width)  # Flatten to [n_way*n_samples, channels, height, width]
        else:
            x_reshaped = x
        
        # Fungsi hook internal untuk menangkap output
        def hook_fn(module, input, output):
            nonlocal features
            if isinstance(output, tuple):
                output = output[0]  # Beberapa model mengembalikan tuple
            features = output
            
            # Flatten if needed, but maintain batch dimension
            # (Ratakan jika perlu, tapi pertahankan dimensi batch)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
        
        # Register hook to the appropriate layer based on model type
        # (Daftarkan hook ke layer yang sesuai berdasarkan tipe model)
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
        # (Jalankan forward pass hanya pada ekstraktor fitur atau backbone)
        try:
            if hasattr(self.model, 'feature'):
                _ = self.model.feature(x_reshaped)
            elif hasattr(self.model, 'feature_extractor'):
                _ = self.model.feature_extractor(x_reshaped)
            else:
                # Only use backbone as fallback (Gunakan backbone sebagai cadangan)
                _ = self.model.backbone(x_reshaped)
        except Exception as e:
            handle.remove()  # Clean up hook before re-raising (Bersihkan hook jika error)
            raise RuntimeError(f"Error during feature extraction: {e}")
        
        # Remove the hook (Hapus hook setelah selesai)
        handle.remove()
        
        return features
        
    def reduce_dimensions(self, method='tsne', n_components=2, **kwargs):
        """
        Reduce dimensions for visualization
        (Reduksi dimensi untuk keperluan visualisasi)
        
        Args:
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            n_components: Number of dimensions to reduce to (2 or 3)
            **kwargs: Additional arguments for the dimensionality reduction method
            
        Returns:
            embeddings: Reduced dimensionality embeddings (Embedding hasil reduksi)
        """
        # Pilih metode reduksi dimensi
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, **kwargs)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
        # Lakukan fit dan transform pada fitur
        embeddings = reducer.fit_transform(self.features)
        return embeddings
        
    def visualize(self, embeddings=None, labels=None, method='tsne', 
                  interactive=True, title=None, save_path=None, **kwargs):
        """
        Visualize the embeddings
        (Memvisualisasikan embedding)
        
        Args:
            embeddings: Pre-computed embeddings (if None, will compute using reduce_dimensions)
            labels: Labels for coloring (if None, will use self.labels)
            method: Method used for dimensionality reduction (for title)
            interactive: Whether to use plotly for interactive visualization (True=Plotly, False=Matplotlib)
            title: Title for the plot (Judul plot)
            save_path: Path to save the visualization (Path penyimpanan file)
            **kwargs: Additional arguments for plotting
            
        Returns:
            fig: The figure object
        """
        if embeddings is None:
            # Hitung embedding jika belum tersedia
            embeddings = self.reduce_dimensions(method=method, **kwargs)
            
        if labels is None:
            labels = self.labels
            
        if title is None:
            title = f"Feature Space Visualization ({method.upper()})"
            
        # Create a DataFrame for easy plotting (Buat DataFrame untuk memudahkan plotting)
        df = pd.DataFrame()
        df['x'] = embeddings[:, 0]
        df['y'] = embeddings[:, 1]
        if embeddings.shape[1] > 2:
            df['z'] = embeddings[:, 2]
            
        df['label'] = labels
        
        # Convert label to string for better formatting (Konversi label ke string)
        df['label'] = df['label'].astype(str)
            
        if interactive:
            # Use plotly for interactive visualization (Gunakan Plotly untuk visualisasi interaktif)
            if embeddings.shape[1] == 2:
                fig = px.scatter(df, x='x', y='y', color='label', 
                                 title=title, hover_data=['label'])
            else:  # 3D
                fig = px.scatter_3d(df, x='x', y='y', z='z', color='label',
                                   title=title, hover_data=['label'])
                
            # Update layout for better appearance (Perbaiki tampilan)
            fig.update_layout(
                legend_title="Class",
                template="plotly_white",
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            # Save if requested (Simpan jika diminta)
            if save_path:
                fig.write_html(save_path)
                
            return fig
        else:
            # Use matplotlib for static visualization (Gunakan Matplotlib untuk visualisasi statis)
            plt.figure(figsize=(10, 8))
            
            if embeddings.shape[1] == 2:
                # 2D plot
                sns.scatterplot(data=df, x='x', y='y', hue='label', palette='viridis', **kwargs)
            else:
                # 3D plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Get unique labels and assign colors (Ambil label unik dan tentukan warna)
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


    
    def compute_gradcam(self, x, target_class_idx, target_layer=None):
        """
        Compute Grad-CAM heatmap for a single image x.
        (Menghitung Grad-CAM heatmpa untuk satu gambar x)
        
        Args:
            x: Input tensor [1, C, H, W]
            target_class_idx: The class index to visualize (Indeks kelas target untuk visualisasi)
            target_layer: The layer to hook (Layer yang akan di-hook)
        
        Returns:
            heatmap: Numpy array (H, W) in range [0, 1]
        """
        self.model.eval()
        x = x.to(self.device).requires_grad_(True)
        
        # 1. Define hooks (Definisikan hooks untuk menangkap gradien dan aktivasi)
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activations = output
            
        # 2. Register hooks (Daftarkan hooks)
        # Identify target layer if not provided
        if target_layer is None:
            # Fallback for Generic/ResNet/Conv4
            if hasattr(self.model, 'feature'):
                # access the underlying sequential
                pass 
                
        # Hardcoding search for last conv for standard backbones
        target_module = None
        
        # Try to find backbone (Cari backbone model)
        backbone = None
        if hasattr(self.model, 'feature'):
             backbone = self.model.feature
        elif hasattr(self.model, 'backbone'):
             backbone = self.model.backbone
             
        if backbone:
            # Walk and find last conv (Telusuri untuk menemukan layer Conv2d terakhir)
            for name, module in backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_module = module
        
        if target_module is None:
             print("Warning: Could not automatically find Conv2d layer for Grad-CAM.")
             return None

        handle_f = target_module.register_forward_hook(forward_hook)
        handle_b = target_module.register_backward_hook(backward_hook)
        
        # 3. Forward Pass
        # Untuk Grad-CAM pada Few-Shot, kita asumsikan struktur standar
        try:
             logits = self.model.set_forward(x)
             # logits shape [n_query_total, n_way]
             
             # Zero grads
             self.model.zero_grad()
             
             # Target score (Targetkan skor spesifik)
             score = logits[0, target_class_idx] 
             score.backward()
             
             # 4. Generate CAM
             # Global Average Pooling of Gradients
             weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
             cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
             cam = torch.nn.functional.relu(cam)
             
             # Upsample to image size (Sesuaikan ukuran heatmap dengan ukuran gambar asli)
             if x.shape[2:] != cam.shape[2:]:
                 cam = torch.nn.functional.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
                 
             cam = cam.view(x.shape[2], x.shape[3]).detach().cpu().numpy()
             
             # Normalize (Normalisasi nilai ke [0, 1])
             cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
             
        except Exception as e:
            print(f"GradCAM Failed: {e}")
            cam = None
            
        finally:
            # Lepaskan hooks
            handle_f.remove()
            handle_b.remove()
            
        return cam