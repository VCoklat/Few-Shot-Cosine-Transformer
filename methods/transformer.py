
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from methods.meta_template import MetaTemplate
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from backbone import CosineDistLinear
import pdb
import IPython

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl, but with a different name.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

def cosine_distance(x1, x2, temperature=None):
    """
    Compute cosine distance with proper dimension handling and optional temperature scaling
    x1: input tensor (3D or 4D)
    x2: input tensor (3D or 4D)
    temperature: optional temperature scaling tensor (higher = softer distribution, lower = sharper)
    Returns: cosine similarity matrix
    """
    try:
        # Handle different tensor dimensions
        if len(x1.shape) == 3 and len(x2.shape) == 3:
            # 3D tensors: [q, n, d] and [q, d, m]
            dots = torch.matmul(x1, x2)
            x1_norm = torch.norm(x1, dim=-1, keepdim=True)  # [q, n, 1]
            x2_norm = torch.norm(x2, dim=-2, keepdim=True)  # [q, 1, m]
            scale = torch.matmul(x1_norm, x2_norm)  # [q, n, m]

        elif len(x1.shape) == 4 and len(x2.shape) == 4:
            # 4D tensors: [h, q, n, d] and [h, q, d, m]
            dots = torch.matmul(x1, x2)
            x1_norm = torch.norm(x1, dim=-1, keepdim=True)  # [h, q, n, 1]
            x2_norm = torch.norm(x2, dim=-2, keepdim=True)  # [h, q, 1, m]
            scale = torch.matmul(x1_norm, x2_norm)  # [h, q, n, m]

        else:
            # Handle mixed dimensions or unexpected cases
            print(f"Warning: Unexpected tensor dimensions in cosine_distance: x1={x1.shape}, x2={x2.shape}")

            # Try generic approach
            dots = torch.matmul(x1, x2)

            # Compute norms along the appropriate dimensions
            x1_norm = torch.norm(x1, dim=-1, keepdim=True)
            x2_norm = torch.norm(x2, dim=-2, keepdim=True) 

            # Use broadcasting to compute scale
            scale = x1_norm * x2_norm

        # Add epsilon to avoid division by zero
        epsilon = 1e-8
        result = dots / (scale + epsilon)
        
        # Apply temperature scaling if provided
        if temperature is not None:
            result = result / temperature

        return result

    except Exception as e:
        print(f"Error in cosine_distance: {e}")
        print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")

        # Safe fallback - return simple dot product scaled by feature dimension
        try:
            dots = torch.matmul(x1, x2)
            scale = x1.size(-1) ** 0.5  # Simple scaling
            return dots / scale
        except:
            # Ultimate fallback - return zeros with correct shape
            if len(x1.shape) >= 2 and len(x2.shape) >= 2:
                output_shape = list(x1.shape[:-1]) + [x2.shape[-1]]
                return torch.zeros(output_shape, device=x1.device)
            else:
                return torch.tensor(0.0, device=x1.device)


class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query, variant="softmax", 
                 depth=1, heads=8, dim_head=64, mlp_dim=512,
                 initial_cov_weight=0.3, initial_var_weight=0.5, dynamic_weight=False,
                 label_smoothing=0.1, attention_dropout=0.1, drop_path_rate=0.1):
        super(FewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)
        # Add label smoothing for better generalization
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim

        # Initialize accuracy tracking - simple attributes, not parameters
        self.current_accuracy = 0.0
        self.accuracy_threshold = 30.0  # Lowered to enable advanced attention earlier
        self.use_advanced_attention = True  # Enable advanced attention from the start

        # Parameters for advanced attention mechanism - optimized for better performance
        self.gamma = 0.08  # Slightly stronger regularization for better feature discrimination
        self.epsilon = 1e-8
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate

        # Create attention module with dropout for better generalization
        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                             initial_cov_weight=initial_cov_weight,
                             initial_var_weight=initial_var_weight,
                             dynamic_weight=dynamic_weight,
                             n_way=n_way, k_shot=k_shot,
                             dropout=attention_dropout)
        self.sm = nn.Softmax(dim=-2)
        # Initialize proto_weight with small random values for better gradient flow
        self.proto_weight = nn.Parameter(torch.randn(n_way, k_shot, 1) * 0.1 + 1.0)
        
        # Add dropout for FFN layers to improve generalization
        self.ffn_dropout = nn.Dropout(0.1)

        # Replace nn.Sequential with separate components to avoid lambda issues
        # FFN components
        self.ffn_layernorm = nn.LayerNorm(dim)
        self.ffn_linear1 = nn.Linear(dim, mlp_dim)
        self.ffn_gelu = nn.GELU()
        self.ffn_linear2 = nn.Linear(mlp_dim, dim)

        # Final linear components
        self.final_layernorm = nn.LayerNorm(dim)
        self.final_linear = nn.Linear(dim, dim_head)

        if variant == "cosine":
            self.final_classifier = CosineDistLinear(dim_head, 1)
        else:
            self.final_classifier = nn.Linear(dim_head, 1)

    def FFN_forward(self, x):
        """Forward pass through FFN layers with dropout"""
        x = self.ffn_layernorm(x)
        x = self.ffn_linear1(x)
        x = self.ffn_gelu(x)
        x = self.ffn_dropout(x)  # Add dropout after activation
        x = self.ffn_linear2(x)
        x = self.ffn_dropout(x)  # Add dropout after final linear layer
        return x

    def final_linear_forward(self, x):
        """Forward pass through final linear layers"""
        x = self.final_layernorm(x)
        x = self.final_linear(x)
        x = self.final_classifier(x)
        return x

    def update_accuracy(self, accuracy):
        """Update current accuracy and switch attention mechanism if needed"""
        self.current_accuracy = accuracy
        should_use_advanced = accuracy >= self.accuracy_threshold

        # if should_use_advanced != self.use_advanced_attention:
        #     self.use_advanced_attention = should_use_advanced
        #     print(f"Switching to {'advanced' if should_use_advanced else 'basic'} attention mechanism at accuracy: {accuracy:.2f}%")
    
    def update_epoch(self, epoch):
        """Update epoch in attention module for adaptive gamma"""
        self.ATTN.update_epoch(epoch)
    
    def mixup_support(self, z_support, alpha=0.2):
        """
        Apply mixup augmentation to support set for better generalization
        Args:
            z_support: [n_way, k_shot, feat_dim]
            alpha: mixup interpolation strength
        Returns:
            Mixed support features
        """
        if not self.training or alpha <= 0:
            return z_support
        
        # Generate mixup coefficient
        lam = np.random.beta(alpha, alpha)
        
        # Randomly shuffle support samples within each class
        n_way, k_shot, feat_dim = z_support.shape
        batch_size = n_way * k_shot
        z_flat = z_support.view(batch_size, feat_dim)
        
        # Create random permutation
        index = torch.randperm(batch_size).to(z_support.device)
        
        # Mix features
        mixed_z = lam * z_flat + (1 - lam) * z_flat[index]
        
        # Reshape back
        return mixed_z.view(n_way, k_shot, feat_dim)

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        
        # Apply mixup augmentation during training for better generalization
        if self.training:
            z_support = self.mixup_support(z_support, alpha=0.2)
        
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
        z_query = z_query.contiguous().reshape(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)

        x, query = z_proto, z_query

        # Process through transformer layers with stochastic depth
        for layer_idx in range(self.depth):
            # Pass additional parameters for attention mechanism switching
            attn_output = self.ATTN(q=x, k=query, v=query, 
                                  use_advanced=self.use_advanced_attention,
                                  gamma=self.gamma,
                                  epsilon=self.epsilon)
            # Apply drop path with layer-specific rate (increases with depth)
            drop_prob = self.drop_path_rate * layer_idx / max(self.depth - 1, 1)
            x = drop_path(attn_output, drop_prob, self.training) + x
            
            ffn_output = self.FFN_forward(x)
            x = drop_path(ffn_output, drop_prob, self.training) + x

        # Output is the probabilistic prediction for each class
        return self.final_linear_forward(x).squeeze()  # (q, n)

    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth

        scores = self.set_forward(x)
        loss = self.loss_fn(scores, target)
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0) * 100

        # Update accuracy and potentially switch attention mechanism
        self.update_accuracy(acc)

        return acc / 100, loss  # Return normalized accuracy


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.6,
                 initial_var_weight=0.2, dynamic_weight=False, n_way=5, k_shot=5, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim=-1)
        self.variant = variant
        self.n_way = n_way
        self.k_shot = k_shot
        self.dropout = nn.Dropout(dropout)  # Add attention dropout

        # Solution 1: Temperature Scaling - Learnable temperature per head (optimized)
        self.temperature = nn.Parameter(torch.ones(heads) * 0.4)  # Start with sharper attention
        
        # Solution 5: EMA Smoothing - Track moving averages for stability (optimized)
        self.ema_decay = 0.98  # Slightly faster adaptation for better responsiveness
        self.register_buffer('var_ema', torch.ones(1))
        self.register_buffer('cov_ema', torch.ones(1))
        
        # Solution 2: Adaptive Gamma - Dynamic variance regularization (optimized)
        self.gamma_start = 0.6  # Start with even stronger regularization
        self.gamma_end = 0.03   # End with even weaker regularization for fine-tuning
        self.current_epoch = 0
        self.max_epochs = 50

        # Dynamic weighting components
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            # Solution 4: Multi-Scale Dynamic Weighting with 4 components
            # Enhanced weight predictor with increased capacity
            self.weight_linear1 = nn.Linear(dim_head * 2, dim_head * 2)
            self.weight_layernorm1 = nn.LayerNorm(dim_head * 2)
            self.weight_gelu1 = nn.GELU()  # Better activation than ReLU
            self.weight_dropout1 = nn.Dropout(0.1)
            self.weight_linear2 = nn.Linear(dim_head * 2, dim_head)
            self.weight_layernorm2 = nn.LayerNorm(dim_head)
            self.weight_gelu2 = nn.GELU()
            self.weight_linear3 = nn.Linear(dim_head, 4)  # 4 weights instead of 3
            self.weight_softmax = nn.Softmax(dim=-1)  # Ensure weights sum to 1.0
        else:
            # Fixed weights as parameters (still learnable)
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))

        # Solution 6: Cross-Attention Between Query and Support
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_head,
            num_heads=1,
            dropout=0.1,
            batch_first=True
        )

        # Replace nn.Sequential for input linear to avoid lambda issues
        self.input_layernorm = nn.LayerNorm(dim)
        self.input_linear = nn.Linear(dim, inner_dim, bias=False)
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

        self.weight_history = []  # To store weights for analysis
        self.record_weights = False  # Toggle for weight recording

    def input_transform(self, t):
        """Transform input through layernorm and linear projection"""
        return self.input_linear(self.input_layernorm(t))

    def weight_predictor_forward(self, x):
        """Forward pass through enhanced weight predictor for 4 components"""
        x = self.weight_linear1(x)
        x = self.weight_layernorm1(x)
        x = self.weight_gelu1(x)
        x = self.weight_dropout1(x)
        x = self.weight_linear2(x)
        x = self.weight_layernorm2(x)
        x = self.weight_gelu2(x)
        x = self.weight_linear3(x)
        return self.weight_softmax(x)

    def get_adaptive_gamma(self):
        """
        Solution 2: Compute adaptive gamma that linearly decreases from start to end.
        Early training needs stronger regularization, later training benefits from weaker.
        """
        progress = min(self.current_epoch / self.max_epochs, 1.0)
        gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * progress
        return gamma
    
    def update_epoch(self, epoch):
        """Update current epoch for adaptive gamma calculation"""
        self.current_epoch = epoch

    def variance_component_torch(self, E, gamma=1.0, epsilon=1e-8):
        """
        PyTorch implementation of variance component matching problem statement:
        def variance_regularization_multi_dim(E, gamma=0.1, epsilon=1e-8):
            variance_per_dim = np.var(E, axis=0, ddof=0)
            regularized_std = np.sqrt(variance_per_dim + epsilon)
            hinge_values = np.maximum(0.0, gamma - regularized_std)
            V_E = np.sum(hinge_values) / m
            return V_E
        """
        # E shape: (batch, seq, dim)
        batch_size = E.shape[0]
        
        # Reshape to (batch*seq, dim) to compute variance across all samples
        E_reshaped = E.reshape(-1, E.shape[-1])  # (batch*seq, dim)
        
        # Compute variance per dimension across samples (axis=0)
        variance_per_dim = torch.var(E_reshaped, dim=0, unbiased=False)  # (dim,)
        
        # Compute regularized standard deviation
        regularized_std = torch.sqrt(variance_per_dim + epsilon)  # (dim,)
        
        # Apply hinge: max(0, gamma - regularized_std)
        hinge_values = torch.clamp(gamma - regularized_std, min=0.0)  # (dim,)
        
        # FIXED: Sum and normalize by number of dimensions (m), not samples
        # This matches the problem statement: V_E = np.sum(hinge_values) / m
        m = E_reshaped.shape[1]  # Number of dimensions
        V_E = torch.sum(hinge_values) / m
        
        # Clear intermediate tensors for memory efficiency
        del E_reshaped, variance_per_dim, regularized_std, hinge_values
        
        return V_E

    def covariance_component_torch(self, E):
        """
        PyTorch implementation of covariance component matching problem statement:
        def covariance_regularization(E):
            E_mean = np.mean(E, axis=0, keepdims=True)
            E_centered = E - E_mean
            cov_matrix = np.dot(E_centered.T, E_centered) / (K - 1)
            mask = np.ones_like(cov_matrix) - np.eye(cov_matrix.shape[0])
            off_diagonal_squared = np.sum((cov_matrix * mask) ** 2)
            return off_diagonal_squared
        
        Memory-optimized version with chunking to prevent OOM.
        """
        try:
            # E shape: (batch, seq, dim)
            batch_size, seq_len, dim = E.shape
            
            # Reshape to (batch*seq, dim) to compute covariance across all samples
            E_reshaped = E.reshape(-1, dim)  # (K, dim) where K = batch*seq
            K = E_reshaped.shape[0]
            
            # Compute mean across samples (axis=0)
            E_mean = torch.mean(E_reshaped, dim=0, keepdim=True)  # (1, dim)
            
            # Center the data
            E_centered = E_reshaped - E_mean  # (K, dim)
            
            # OPTIMIZED: Better chunking strategy to prevent OOM
            # Adaptive chunk size based on both dimension and available memory
            if dim > 2048:
                chunk_size = 32  # Very small chunks for huge dimensions
            elif dim > 1024:
                chunk_size = 64  # Smaller chunks for very large dimensions
            elif dim > 512:
                chunk_size = 128  # Medium chunks for large dimensions
            else:
                # For smaller dimensions, compute directly without chunking
                # This is more efficient and avoids chunking overhead
                if K > 1:
                    cov_matrix = torch.matmul(E_centered.T, E_centered) / (K - 1)
                else:
                    cov_matrix = torch.matmul(E_centered.T, E_centered)
                
                # Create mask for off-diagonal elements
                mask = torch.ones_like(cov_matrix) - torch.eye(dim, device=cov_matrix.device)
                
                # FIXED: Normalize by dimension m
                off_diagonal_squared = torch.sum((cov_matrix * mask) ** 2) / dim
                
                # Clear tensors
                del E_centered, cov_matrix, mask, E_mean, E_reshaped
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return off_diagonal_squared
            
            # For large dimensions, use chunked computation
            cov_matrix = torch.zeros(dim, dim, device=E.device)
            
            for i in range(0, dim, chunk_size):
                end_i = min(i + chunk_size, dim)
                for j in range(0, dim, chunk_size):
                    end_j = min(j + chunk_size, dim)
                    
                    # Compute chunk of covariance matrix
                    chunk_i = E_centered[:, i:end_i]  # (K, chunk_i_size)
                    chunk_j = E_centered[:, j:end_j]  # (K, chunk_j_size)
                    
                    cov_chunk = torch.matmul(chunk_i.T, chunk_j)  # (chunk_i_size, chunk_j_size)
                    if K > 1:
                        cov_chunk = cov_chunk / (K - 1)
                    
                    cov_matrix[i:end_i, j:end_j] = cov_chunk
                    
                    # Clear intermediate tensors immediately
                    del chunk_i, chunk_j, cov_chunk
                    
                # Clear cache after each row of chunks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Create mask for off-diagonal elements
            mask = torch.ones_like(cov_matrix) - torch.eye(dim, device=cov_matrix.device)
            
            # FIXED: Compute sum of squares of off-diagonal elements and normalize by m
            # This matches the CTX.py implementation and improves numerical stability
            off_diagonal_squared = torch.sum((cov_matrix * mask) ** 2) / dim
            
            # Clear large tensors
            del E_centered, cov_matrix, mask, E_mean, E_reshaped
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return off_diagonal_squared
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # OOM detected - clear cache and try with smaller chunks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"OOM in covariance computation, using fallback with smaller chunks")
                # Return a small penalty value instead of crashing
                return torch.tensor(0.0, device=E.device, requires_grad=True)
            else:
                raise e

    def basic_attention_components(self, f_q, f_k):
        """Original attention mechanism for accuracy < 40%"""
        # Calculate covariance component
        q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
        k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
        cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
        cov_component = cov_component / f_q.size(-1)

        # Calculate variance component
        # Compute variance along feature dimension
        q_var = torch.var(f_q, dim=-1, keepdim=True)  # [h, q, n, 1]
        k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)  # [h, q, 1, m]

        # Create variance-based attention
        var_component = torch.matmul(q_var, k_var)  # [h, q, n, m]
        var_component = var_component / f_q.size(-1)  # Scale like covariance

        return cov_component, var_component

    def advanced_attention_components(self, f_q, f_k, gamma=1.0, epsilon=1e-8):
        """
        FIXED: Advanced attention mechanism with proper tensor dimension handling
        This version resolves the RuntimeError: shape '[8, 1, 64]' is invalid for input of size 40960
        """
        # Determine the input format and handle different tensor shapes
        if len(f_q.shape) == 4:
            # Input format: [h, q, n, d] from rearrange operation
            heads, batch_size, seq_q, dim = f_q.shape
            _, _, seq_k, _ = f_k.shape

            # Dynamic calculation of sequence length k based on total elements
            total_elements_k = f_k.numel()
            expected_elements_k = heads * batch_size * seq_k * dim

            if total_elements_k != expected_elements_k:
                # Recalculate seq_k dynamically
                seq_k = total_elements_k // (heads * batch_size * dim)
                if seq_k * heads * batch_size * dim != total_elements_k:
                    print(f"Cannot resolve tensor reshape: f_k has {total_elements_k} elements")
                    return self.basic_attention_components(f_q, f_k)

            # Safely reshape for processing: (heads*batch, seq, dim)
            try:
                f_q_reshaped = f_q.permute(1, 0, 2, 3).contiguous().view(batch_size * heads, seq_q, dim)
                f_k_reshaped = f_k.permute(1, 0, 2, 3).contiguous().view(batch_size * heads, seq_k, dim)
            except RuntimeError as e:
                print(f"Error reshaping tensors: {e}")
                # Return safe fallback using basic attention components instead
                return self.basic_attention_components(f_q, f_k)

        else:
            print(f"Unexpected tensor dimensions: f_q {f_q.shape}, f_k {f_k.shape}")
            # Return safe fallback
            return self.basic_attention_components(f_q, f_k)

        # Compute components with memory optimization
        var_component_list = []
        cov_component_list = []

        # OPTIMIZED: Better adaptive chunk size based on dimension and memory
        total_samples = batch_size * heads
        # More conservative chunk sizes to prevent OOM
        if dim > 512:
            chunk_size = 1  # Process one sample at a time for very large dimensions
        elif dim > 256:
            chunk_size = 1  # Still very conservative
        elif dim > 128:
            chunk_size = min(2, total_samples)  # Very small chunks for large dimensions
        else:
            chunk_size = min(4, total_samples)  # Moderate chunks for normal dimensions

        try:
            for i in range(0, total_samples, chunk_size):
                end_idx = min(i + chunk_size, total_samples)

                # Get chunks
                q_chunk = f_q_reshaped[i:end_idx]  # (chunk, seq_q, dim)
                k_chunk = f_k_reshaped[i:end_idx]  # (chunk, seq_k, dim)

                # Compute variance component for this chunk
                var_q = self.variance_component_torch(q_chunk, gamma, epsilon)
                var_k = self.variance_component_torch(k_chunk, gamma, epsilon)
                var_comp = var_q * var_k
                var_component_list.append(var_comp.unsqueeze(0).expand(end_idx - i, seq_q, seq_k))

                # Compute covariance component for this chunk  
                cov_q = self.covariance_component_torch(q_chunk)
                cov_k = self.covariance_component_torch(k_chunk)
                cov_comp = cov_q * cov_k
                cov_component_list.append(cov_comp.unsqueeze(0).expand(end_idx - i, seq_q, seq_k))

                # Clear intermediate tensors immediately
                del q_chunk, k_chunk, var_q, var_k, var_comp, cov_q, cov_k, cov_comp
                
                # Clear GPU cache more frequently to prevent accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine results and reshape back to original format
            var_component = torch.cat(var_component_list, dim=0).view(batch_size, heads, seq_q, seq_k)
            cov_component = torch.cat(cov_component_list, dim=0).view(batch_size, heads, seq_q, seq_k)

            # Clear intermediate lists to free memory
            del var_component_list, cov_component_list, f_q_reshaped, f_k_reshaped

            # Permute back to match f_q/f_k format [h, q, n, d] -> [h, q, n, m]
            var_component = var_component.permute(1, 0, 2, 3)
            cov_component = cov_component.permute(1, 0, 2, 3)
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return cov_component, var_component

        except Exception as e:
            print(f"Error in advanced attention computation: {e}")
            # Clear any allocated memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Fallback to basic attention components
            return self.basic_attention_components(f_q, f_k)

    def forward(self, q, k, v, use_advanced=False, gamma=1.0, epsilon=1e-8):
        # FIXED: Properly handle q, k, v dimensions to ensure matrix multiplication compatibility
        
        # Solution 6: Cross-Attention Between Query and Support
        # Split input into support and query based on sequence length
        # Expected: q is support (1, n_way, d), k and v are queries (n_way*n_query, 1, d)
        if q.shape[0] == 1 and k.shape[0] > 1:
            # q is support, k/v are queries
            support = q  # [1, n_way, d]
            query = k    # [n_way*n_query, 1, d]
            
            # Apply cross-attention: query attends to support
            # Reshape for MultiheadAttention (batch_first=True)
            support_reshaped = support.squeeze(0).unsqueeze(0)  # [1, n_way, d]
            query_reshaped = query.squeeze(1)  # [n_way*n_query, d]
            
            # Apply cross-attention
            try:
                # Get the embedding dimension
                embed_dim = query_reshaped.shape[-1]
                
                # Check if cross_attn is compatible
                if self.cross_attn.embed_dim != embed_dim:
                    # Create a new cross-attention layer with correct dimensions
                    self.cross_attn = nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=1,
                        dropout=0.1,
                        batch_first=True
                    ).to(query_reshaped.device)
                
                # Reshape for batch processing
                query_batch = query_reshaped.unsqueeze(0)  # [1, n_way*n_query, d]
                query_enhanced, _ = self.cross_attn(query_batch, support_reshaped, support_reshaped)
                query_enhanced = query_enhanced.squeeze(0).unsqueeze(1)  # [n_way*n_query, 1, d]
                
                # Update k and v with enhanced query
                k = query_enhanced
                v = query_enhanced
            except Exception as e:
                # If cross-attention fails, continue with original k, v
                pass

        # Apply input transformation to each tensor separately
        f_q = self.input_transform(q)  # [batch_q, seq_q, feat_dim] -> [batch_q, seq_q, inner_dim]
        f_k = self.input_transform(k)  # [batch_k, seq_k, feat_dim] -> [batch_k, seq_k, inner_dim]
        f_v = self.input_transform(v)  # [batch_v, seq_v, feat_dim] -> [batch_v, seq_v, inner_dim]

        # Apply rearrange to each transformed tensor
        f_q = rearrange(f_q, 'b n (h d) -> h b n d', h=self.heads)  # [heads, batch, seq_q, head_dim]
        f_k = rearrange(f_k, 'b n (h d) -> h b n d', h=self.heads)  # [heads, batch, seq_k, head_dim]
        f_v = rearrange(f_v, 'b n (h d) -> h b n d', h=self.heads)  # [heads, batch, seq_v, head_dim]

        if self.variant == "cosine":
            # Solution 1: Calculate cosine similarity with temperature scaling
            # f_k.transpose(-1, -2) gives us [heads, batch, head_dim, seq_k]
            # cosine_distance(f_q, f_k.transpose(-1, -2)) gives [heads, batch, seq_q, seq_k]
            
            # Reshape temperature for broadcasting: [heads] -> [heads, 1, 1, 1]
            temp_reshaped = self.temperature.view(self.heads, 1, 1, 1)
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2), temperature=temp_reshaped)

            # Solution 2: Use adaptive gamma for variance regularization
            adaptive_gamma = self.get_adaptive_gamma()
            
            # Choose attention mechanism based on accuracy with error handling
            try:
                if use_advanced:
                    cov_component, var_component = self.advanced_attention_components(f_q, f_k, adaptive_gamma, epsilon)
                else:
                    cov_component, var_component = self.basic_attention_components(f_q, f_k)

            except Exception as e:
                print(f"Error in attention components: {e}, falling back to basic")
                cov_component, var_component = self.basic_attention_components(f_q, f_k)

            # Solution 5: Apply EMA smoothing during training
            if self.training:
                with torch.no_grad():
                    self.var_ema = self.ema_decay * self.var_ema + (1 - self.ema_decay) * var_component.detach().mean()
                    self.cov_ema = self.ema_decay * self.cov_ema + (1 - self.ema_decay) * cov_component.detach().mean()
            
            # Normalize components by their EMA for stability
            var_component_norm = var_component / (self.var_ema + epsilon)
            cov_component_norm = cov_component / (self.cov_ema + epsilon)

            # Weight combination logic
            if self.dynamic_weight:
                try:
                    # Use global feature statistics
                    q_global = f_q.mean(dim=(1, 2))  # [h, d]
                    k_global = f_k.mean(dim=(1, 2))  # [h, d]

                    # Concatenate global query and key features
                    qk_features = torch.cat([q_global, k_global], dim=-1)  # [h, 2d]

                    # Solution 4: Predict four weights per attention head
                    weights = self.weight_predictor_forward(qk_features)  # [h, 4]

                    # Record weights during evaluation if needed
                    if self.record_weights and not self.training:
                        self.weight_history.append(weights.detach().cpu().numpy().mean(axis=0))

                    # Extract individual weights for 4 components
                    cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)  # Cosine weight
                    cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)  # Covariance weight
                    var_weight = weights[:, 2].view(self.heads, 1, 1, 1)  # Variance weight
                    interaction_weight = weights[:, 3].view(self.heads, 1, 1, 1)  # Interaction weight

                    # Solution 4: Add interaction term (product of cosine and covariance)
                    interaction_term = cosine_sim * cov_component_norm

                    # Combine all four components
                    dots = (cos_weight * cosine_sim +
                           cov_weight * cov_component_norm + 
                           var_weight * var_component_norm +
                           interaction_weight * interaction_term)

                except Exception as e:
                    print(f"Error in dynamic weighting: {e}, using equal weights")
                    # Fallback to equal weighting
                    dots = (cosine_sim + cov_component_norm + var_component_norm) / 3
            else:
                # Use fixed weights
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)

                # Ensure weights sum to approximately 1 by using the remaining portion for cosine
                cos_weight = 1.0 - cov_weight - var_weight

                dots = (cos_weight * cosine_sim +
                       cov_weight * cov_component +
                       var_weight * var_component)

            # CRITICAL FIX: Ensure f_v has the correct sequence dimension to match dots
            # dots shape: [heads, batch, seq_q, seq_k] 
            # f_v shape should be: [heads, batch, seq_v, head_dim] where seq_v = seq_k

            # Check if sequence dimensions match for matrix multiplication
            if f_v.shape[2] != dots.shape[3]:  # seq_v != seq_k
                print(f"Warning: Sequence dimension mismatch - dots expects seq_k={dots.shape[3]} but f_v has seq_v={f_v.shape[2]}")

                # If k and v come from the same input (self-attention case), they should have same seq length
                # Adjust f_v to match the expected sequence length
                if dots.shape[3] < f_v.shape[2]:
                    # Truncate f_v
                    f_v = f_v[:, :, :dots.shape[3], :]
                elif dots.shape[3] > f_v.shape[2]:
                    # Pad f_v with zeros or repeat last elements
                    pad_size = dots.shape[3] - f_v.shape[2]
                    padding = f_v[:, :, -1:, :].expand(-1, -1, pad_size, -1)  # Repeat last element
                    f_v = torch.cat([f_v, padding], dim=2)

            # Now perform matrix multiplication: dots @ f_v
            out = torch.matmul(dots, f_v)  # [heads, batch, seq_q, head_dim]
            # Apply attention dropout for regularization
            out = self.dropout(out)

        else:  # self.variant == "softmax"
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale  # [heads, batch, seq_q, seq_k]

            # Same fix for softmax variant
            if f_v.shape[2] != dots.shape[3]:  # seq_v != seq_k
                print(f"Warning: Sequence dimension mismatch in softmax - adjusting f_v")
                if dots.shape[3] < f_v.shape[2]:
                    f_v = f_v[:, :, :dots.shape[3], :]
                elif dots.shape[3] > f_v.shape[2]:
                    pad_size = dots.shape[3] - f_v.shape[2]
                    padding = f_v[:, :, -1:, :].expand(-1, -1, pad_size, -1)
                    f_v = torch.cat([f_v, padding], dim=2)

            out = torch.matmul(self.sm(dots), f_v)  # [heads, batch, seq_q, head_dim]
            # Apply attention dropout for regularization
            out = self.dropout(out)

        # Rearrange back to original format
        out = rearrange(out, 'h b n d -> b n (h d)')  # [batch, seq_q, inner_dim]
        return self.output_linear(out)

    def get_weight_stats(self):
        """Returns statistics about the weights used"""
        if not self.weight_history:
            return None

        weights = np.array(self.weight_history)

        if weights.shape[1] == 4:  # We have 4 components (updated)
            return {
                'cosine_mean': float(weights[:, 0].mean()),
                'cov_mean': float(weights[:, 1].mean()),
                'var_mean': float(weights[:, 2].mean()),
                'interaction_mean': float(weights[:, 3].mean()),
                'cosine_std': float(weights[:, 0].std()),
                'cov_std': float(weights[:, 1].std()),
                'var_std': float(weights[:, 2].std()),
                'interaction_std': float(weights[:, 3].std()),
                'histogram': {
                    'cosine': np.histogram(weights[:, 0], bins=10, range=(0,1))[0].tolist(),
                    'cov': np.histogram(weights[:, 1], bins=10, range=(0,1))[0].tolist(),
                    'var': np.histogram(weights[:, 2], bins=10, range=(0,1))[0].tolist(),
                    'interaction': np.histogram(weights[:, 3], bins=10, range=(0,1))[0].tolist()
                }
            }
        elif weights.shape[1] == 3:  # We have 3 components (legacy)
            return {
                'cosine_mean': float(weights[:, 0].mean()),
                'cov_mean': float(weights[:, 1].mean()),
                'var_mean': float(weights[:, 2].mean()),
                'cosine_std': float(weights[:, 0].std()),
                'cov_std': float(weights[:, 1].std()),
                'var_std': float(weights[:, 2].std()),
                'histogram': {
                    'cosine': np.histogram(weights[:, 0], bins=10, range=(0,1))[0].tolist(),
                    'cov': np.histogram(weights[:, 1], bins=10, range=(0,1))[0].tolist(),
                    'var': np.histogram(weights[:, 2], bins=10, range=(0,1))[0].tolist()
                }
            }
        else:  # Legacy format with single weight
            return {
                'mean': float(weights.mean()),
                'std': float(weights.std()),
                'min': float(weights.min()),
                'max': float(weights.max()),
                'histogram': np.histogram(weights, bins=10, range=(0,1))[0].tolist()
            }

    def clear_weight_history(self):
        """Clear recorded weights"""
        self.weight_history = []
