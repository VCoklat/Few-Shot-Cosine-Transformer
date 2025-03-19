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
import gc
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query, variant="softmax",
                depth=1, heads=8, dim_head=64, mlp_dim=512,
                use_vic_reg=True, weight_variance=25, weight_invariance=25, weight_covariance=1):
        super(FewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        
        # VIC regularization parameters
        self.use_vic_reg = use_vic_reg
        self.weight_variance = weight_variance
        self.weight_invariance = weight_invariance
        self.weight_covariance = weight_covariance
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim

        # Use simplified attention mechanism
        self.ATTN = SimplifiedAttention(dim, heads=heads, dim_head=dim_head, variant=variant)
        
        self.sm = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim))
        
        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head),
            CosineDistLinear(dim_head, 1) if variant == "cosine"
            else nn.Linear(dim_head, 1))
        
    def set_forward(self, x, is_feature=False):

        z_support, z_query = self.parse_feature(x, is_feature)
                
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)                         # (1, n, d)
        
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)                # (q, 1, d)

        x, query = z_proto, z_query
        
        for _ in range(self.depth):
           x = self.ATTN(q = x, k = query, v = query) + x
           x = self.FFN(x) + x
        
        # Output is the probabilistic prediction for each class
        return self.linear(x).squeeze()                                                                # (q, n)
    
    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        
        # Get scores first (reuses feature computation)
        scores = self.set_forward(x, is_feature=False)
        
        # Calculate standard classification loss
        classification_loss = self.loss_fn(scores, target)
        
        # Add VIC regularization if enabled
        if self.use_vic_reg:
            # We'll avoid CPU computations and directly reuse the features
            with torch.no_grad():
                # Extract features without moving to CPU (which causes the shape issue)
                z_support, z_query = self.parse_feature(x, is_feature=False)
                
                # Reshape support features for regularization
                support_features = z_support.view(-1, z_support.size(-1))
                
                # Compute prototypes
                z_support_reshaped = z_support.contiguous().view(self.n_way, self.k_shot, -1)
                prototypes = (z_support_reshaped * self.sm(self.proto_weight)).sum(1)
                
                # Process variance in chunks if needed (remaining code stays the same)
                if support_features.shape[0] > 50:  # If we have a large number of support features
                    chunk_size = 50
                    
                    std_loss_list = []
                    
                    # Process variance in chunks
                    for i in range(0, support_features.shape[0], chunk_size):
                        end = min(i + chunk_size, support_features.shape[0])
                        chunk = support_features[i:end]
                        std_chunk = torch.sqrt(chunk.var(dim=0) + 1e-4)
                        std_loss_chunk = torch.mean(F.relu(1 - std_chunk))
                        std_loss_list.append(std_loss_chunk)
                    
                    std_z_a_loss = sum(std_loss_list) / len(std_loss_list)
                else:
                    std_z_a = torch.sqrt(support_features.var(dim=0) + 1e-4)
                    std_z_a_loss = torch.mean(F.relu(1 - std_z_a))
                
                # Process prototype variance
                std_z_b = torch.sqrt(prototypes.var(dim=0) + 1e-4)
                std_loss = std_z_a_loss + torch.mean(F.relu(1 - std_z_b))
                    
                # Covariance calculation with memory optimizations
                N_a, D = support_features.shape
                N_b, _ = prototypes.shape
                
                # Compute covariance in chunks to save memory
                off_diag_sum = 0
                chunk_size = min(30, N_a + N_b)  # Smaller chunk size
                
                for i in range(0, N_a + N_b, chunk_size):
                    end = min(i + chunk_size, N_a + N_b)
                    
                    # Get the appropriate slices
                    if i < N_a and end <= N_a:
                        chunk = support_features[i:end]
                    elif i < N_a:
                        chunk = torch.cat([support_features[i:], prototypes[:end-(N_a)]]) 
                    else:
                        chunk = prototypes[i-N_a:end-N_a]
                        
                    # Center this chunk
                    chunk = chunk - chunk.mean(dim=0)
                    
                    # Compute partial covariance
                    cov_chunk = (chunk.T @ chunk) / (end - i - 1 + 1e-6)
                    
                    # Remove diagonal and add to sum
                    off_diag_chunk = cov_chunk - torch.diag(torch.diagonal(cov_chunk))
                    off_diag_sum += off_diag_chunk.pow(2).sum()
                    
                    # Free memory
                    del chunk, cov_chunk, off_diag_chunk
                    torch.cuda.empty_cache()
                    
                cov_loss = off_diag_sum / D
                    
                # Combined loss with chunked processing
                loss = classification_loss + \
                       self.weight_variance * std_loss + \
                       self.weight_covariance * cov_loss
        else:
            loss = classification_loss
            
        # Calculate accuracy
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        
        return acc, loss

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.6, initial_var_weight=0.3, dynamic_weight=False):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim = -1)
        self.variant = variant
        
        # Dynamic weighting components
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            # Network to predict the weights based on features (now 3 components)
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.LayerNorm(dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, 3),  # Now predict 3 weights instead of 1
                nn.Softmax(dim=-1)  # Ensure weights sum to 1.0
            )
        else:
            # Fixed weights as parameters (still learnable)
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))
            
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias = False))
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        
        self.weight_history = []  # To store weights for analysis
        self.record_weights = False  # Toggle for weight recording
    
    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) ->  h q n d', h = self.heads), (q, k ,v))    
        
        if self.variant == "cosine":
            # Calculate cosine similarity (invariance component)
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            
            # Calculate covariance component
            q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
            k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
            cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
            cov_component = cov_component / f_q.size(-1)
            
            # Calculate variance component (new)
            # Compute variance along feature dimension
            q_var = torch.var(f_q, dim=-1, keepdim=True)  # [h, q, n, 1]
            k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)  # [h, q, 1, m]
            
            # Create variance-based attention
            var_component = torch.matmul(q_var, k_var)  # [h, q, n, m]
            var_component = var_component / f_q.size(-1)  # Scale like covariance
            
            if self.dynamic_weight:
                # Use global feature statistics
                q_global = f_q.mean(dim=(1, 2))  # [h, d]
                k_global = f_k.mean(dim=(1, 2))  # [h, d]
                
                # Concatenate global query and key features
                qk_features = torch.cat([q_global, k_global], dim=-1)  # [h, 2d]
                
                # Predict three weights per attention head
                weights = self.weight_predictor(qk_features)  # [h, 3]
                
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
        
        else: # self.variant == "softmax" 
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
            weights = np.array(self.weight_history)
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

class SimplifiedAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim=-1)
        self.variant = variant
            
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False)
        )
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
    
    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) ->  h q n d', h=self.heads), (q, k, v))    
        
        if self.variant == "cosine":
            # Use only cosine similarity
            dots = cosine_distance(f_q, f_k.transpose(-1, -2))
            out = torch.matmul(dots, f_v)
        else:  # self.variant == "softmax"
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale            
            out = torch.matmul(self.sm(dots), f_v)
        
        out = rearrange(out, 'h q n d -> q n (h d)')
        return self.output_linear(out)

def cosine_distance(x1, x2):
    '''
    x1      =  [b, h, n, k]
    x2      =  [b, h, k, m]
    output  =  [b, h, n, m]
    '''
    dots = torch.matmul(x1, x2)
    scale = torch.einsum('bhi, bhj -> bhij', 
            (torch.norm(x1, 2, dim = -1), torch.norm(x2, 2, dim = -2)))
    return (dots / scale)
