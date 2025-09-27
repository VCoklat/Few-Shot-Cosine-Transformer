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

class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query, variant="softmax",
                 depth=1, heads=8, dim_head=64, mlp_dim=512,
                 initial_cov_weight=0.3, initial_var_weight=0.5, dynamic_weight=False):
        super(FewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)
        self.loss_fn = nn.CrossEntropyLoss()
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim
        
        # Initialize accuracy tracking - simple attributes, not parameters
        self.current_accuracy = 0.0
        self.accuracy_threshold = 40.0
        self.use_advanced_attention = False
        
        # Parameters for advanced attention mechanism
        self.gamma = 1.0
        self.epsilon = 1e-8
        
        # Create attention module
        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                            initial_cov_weight=initial_cov_weight,
                            initial_var_weight=initial_var_weight,
                            dynamic_weight=dynamic_weight)
        
        self.sm = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        
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
        """Forward pass through FFN layers"""
        x = self.ffn_layernorm(x)
        x = self.ffn_linear1(x)
        x = self.ffn_gelu(x)
        x = self.ffn_linear2(x)
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
        
        if should_use_advanced != self.use_advanced_attention:
            self.use_advanced_attention = should_use_advanced
            print(f"Switching to {'advanced' if should_use_advanced else 'basic'} attention mechanism at accuracy: {accuracy:.2f}%")
        
    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)
        
        x, query = z_proto, z_query
        
        # Process through transformer layers
        for _ in range(self.depth):
            # Pass additional parameters for attention mechanism switching
            attn_output = self.ATTN(q=x, k=query, v=query, 
                                  use_advanced=self.use_advanced_attention,
                                  gamma=self.gamma,
                                  epsilon=self.epsilon)
            x = attn_output + x
            x = self.FFN_forward(x) + x
        
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
                 initial_var_weight=0.2, dynamic_weight=False):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim=-1)
        self.variant = variant
        
        # Dynamic weighting components
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            # Separate components instead of nn.Sequential to avoid recursion
            self.weight_linear1 = nn.Linear(dim_head * 2, dim_head)
            self.weight_layernorm = nn.LayerNorm(dim_head)
            self.weight_relu = nn.ReLU()
            self.weight_linear2 = nn.Linear(dim_head, 3)  # Now predict 3 weights
            self.weight_softmax = nn.Softmax(dim=-1)  # Ensure weights sum to 1.0
        else:
            # Fixed weights as parameters (still learnable)
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))
        
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
        """Forward pass through weight predictor"""
        x = self.weight_linear1(x)
        x = self.weight_layernorm(x)
        x = self.weight_relu(x)
        x = self.weight_linear2(x)
        return self.weight_softmax(x)

    def variance_component_torch(self, E, gamma=1.0, epsilon=1e-8):
        """PyTorch implementation of variance component"""
        # E shape: (batch, seq, dim)
        sigma = torch.sqrt(torch.var(E, dim=1, keepdim=True) + epsilon)  # (batch, 1, dim)
        V = torch.mean(torch.clamp(gamma - sigma, min=0.0))
        return V

    def covariance_component_torch(self, E):
        """PyTorch implementation of covariance component"""
        # E shape: (batch, seq, dim)
        batch_size, seq_len, dim = E.shape
        
        # Process in chunks to avoid OOM
        chunk_size = min(32, batch_size)
        cov_results = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            E_chunk = E[i:end_idx]  # (chunk, seq, dim)
            
            # Compute mean and center the data
            E_mean = torch.mean(E_chunk, dim=1, keepdim=True)  # (chunk, 1, dim)
            centered = E_chunk - E_mean  # (chunk, seq, dim)
            
            # Compute covariance matrix for each sample in chunk
            for j in range(E_chunk.shape[0]):
                centered_sample = centered[j]  # (seq, dim)
                cov = torch.matmul(centered_sample.T, centered_sample) / (seq_len - 1)  # (dim, dim)
                
                # Sum of squares of off-diagonal elements
                off_diag_mask = ~torch.eye(dim, dtype=torch.bool, device=cov.device)
                off_diag_sum = torch.sum(cov[off_diag_mask] ** 2)
                
                # Normalize by dimension
                C = off_diag_sum / dim
                cov_results.append(C)
            
            # Clear intermediate tensors
            del E_chunk, E_mean, centered
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.stack(cov_results).mean()

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
        """Advanced attention mechanism for accuracy >= 40%"""
        batch_size, heads, seq_q, dim = f_q.shape
        _, _, seq_k, _ = f_k.shape
        
        # Process embeddings for variance and covariance components
        # Reshape for processing: (batch*heads, seq, dim)
        f_q_reshaped = f_q.view(batch_size * heads, seq_q, dim)
        f_k_reshaped = f_k.view(batch_size * heads, seq_k, dim)
        
        # Compute components with memory optimization
        var_component_list = []
        cov_component_list = []
        
        # Process in smaller chunks to avoid OOM
        chunk_size = min(8, batch_size * heads)
        
        for i in range(0, batch_size * heads, chunk_size):
            end_idx = min(i + chunk_size, batch_size * heads)
            
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
            
            # Clear intermediate tensors
            del q_chunk, k_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine results
        var_component = torch.cat(var_component_list, dim=0).view(batch_size, heads, seq_q, seq_k)
        cov_component = torch.cat(cov_component_list, dim=0).view(batch_size, heads, seq_q, seq_k)
        
        return cov_component, var_component

    def forward(self, q, k, v, use_advanced=False, gamma=1.0, epsilon=1e-8):
        # Apply input transformation - avoid lambda functions
        f_q = rearrange(self.input_transform(q), 'q n (h d) -> h q n d', h=self.heads)
        f_k = rearrange(self.input_transform(k), 'q n (h d) -> h q n d', h=self.heads)
        f_v = rearrange(self.input_transform(v), 'q n (h d) -> h q n d', h=self.heads)
        
        if self.variant == "cosine":
            # Calculate cosine similarity (invariance component)
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            
            # Choose attention mechanism based on accuracy
            if use_advanced:
                cov_component, var_component = self.advanced_attention_components(f_q, f_k, gamma, epsilon)
            else:
                cov_component, var_component = self.basic_attention_components(f_q, f_k)
            
            # Weight combination logic
            if self.dynamic_weight:
                # Use global feature statistics
                q_global = f_q.mean(dim=(1, 2))  # [h, d]
                k_global = f_k.mean(dim=(1, 2))  # [h, d]
                # Concatenate global query and key features
                qk_features = torch.cat([q_global, k_global], dim=-1)  # [h, 2d]
                # Predict three weights per attention head
                weights = self.weight_predictor_forward(qk_features)  # [h, 3]
                
                # Record weights during evaluation if needed
                if self.record_weights and not self.training:
                    self.weight_history.append(weights.detach().cpu().numpy().mean(axis=0))
                
                # Extract individual weights
                cos_weight = weights[:, 0].view(self.heads, 1, 1, 1)  # Cosine weight
                cov_weight = weights[:, 1].view(self.heads, 1, 1, 1)  # Covariance weight
                var_weight = weights[:, 2].view(self.heads, 1, 1, 1)  # Variance weight
                
                # Combine all three components
                dots = (cos_weight * cosine_sim +
                       cov_weight * cov_component +
                       var_weight * var_component)
            else:
                # Use fixed weights
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                # Ensure weights sum to approximately 1 by using the remaining portion for cosine
                cos_weight = 1.0 - cov_weight - var_weight
                
                dots = (cos_weight * cosine_sim +
                       cov_weight * cov_component +
                       var_weight * var_component)
            
            out = torch.matmul(dots, f_v)
            
        else:  # self.variant == "softmax"
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out = torch.matmul(self.sm(dots), f_v)
        
        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)

    def get_weight_stats(self):
        """Returns statistics about the weights used"""
        if not self.weight_history:
            return None
        
        weights = np.array(self.weight_history)
        if weights.shape[1] == 3:  # We have 3 components
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


def cosine_distance(x1, x2):
    '''
    x1 = [b, h, n, k]
    x2 = [b, h, k, m]
    output = [b, h, n, m]
    '''
    dots = torch.matmul(x1, x2)
    scale = torch.einsum('bhi, bhj -> bhij',
                        (torch.norm(x1, 2, dim=-1), torch.norm(x2, 2, dim=-2)))
    return (dots / scale)
