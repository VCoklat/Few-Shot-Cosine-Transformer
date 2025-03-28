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
                initial_cov_weight=0.8494, initial_var_weight=0.0009, dynamic_weight=True):
        super(FewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim

        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                             initial_cov_weight=initial_cov_weight,
                             initial_var_weight=initial_var_weight,
                             dynamic_weight=dynamic_weight)
        
        self.sm = nn.Softmax(dim = -2)
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
        scores = self.set_forward(x)
        
        loss = self.loss_fn(scores, target)
        predict = torch.argmax(scores, dim = 1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, loss

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight, initial_var_weight, dynamic_weight):
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
        """Clear recorded weight history"""
        self.weight_history = []

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

class DynamicFewShotTransformer(FewShotTransformer):
    """
    A Few-Shot Transformer model with architecture parameters that adapt to each task.
    
    This model dynamically adjusts:
    - Depth (number of layers)
    - Number of attention heads
    - Hidden dimension size
    - MLP dimension size
    
    Based on the task embedding learned from support set features.
    """
    def __init__(self, model_func, n_way, k_shot, n_query, variant="softmax",
                min_depth=1, max_depth=4, 
                min_heads=4, max_heads=12,
                min_dim_head=32, max_dim_head=96,
                min_mlp=256, max_mlp=768,
                initial_cov_weight=0.8494, initial_var_weight=0.0009, 
                dynamic_weight=True):
        
        # Initialize with base parameters, avoiding parent's architecture initialization
        super(FewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.k_shot = k_shot
        self.variant = variant
        dim = self.feat_dim
        
        # Store architecture bounds
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_heads = min_heads
        self.max_heads = max_heads
        self.min_dim_head = min_dim_head
        self.max_dim_head = max_dim_head
        self.min_mlp = min_mlp
        self.max_mlp = max_mlp
        
        # Architecture predictor based on task embedding
        self.arch_predictor = nn.Sequential(
            nn.Linear(dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 outputs: depth, heads, dim_head, mlp_dim
        )
        
        # Create softmax for prototype weighting
        self.sm = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        
        # Build dynamic components with maximum sizes
        # Multiple transformer layers to support variable depth
        self.layers = nn.ModuleList([
            DynamicTransformerLayer(
                dim, max_heads, max_dim_head, max_mlp, variant,
                initial_cov_weight, initial_var_weight, dynamic_weight
            ) for _ in range(max_depth)
        ])
        
        # Dynamic output projection
        self.dynamic_linear = DynamicProjection(dim, max_dim_head, variant)
        
        # Layer importance predictor (for differentiable layer selection)
        self.layer_importance = nn.Parameter(torch.ones(max_depth))
        
        # Architecture metrics for monitoring
        self.arch_history = []
        
    def get_dynamic_architecture(self, support_features):
        """Predict architecture parameters based on task features"""
        # Average support features to get task embedding
        task_embedding = support_features.mean(dim=(0, 1))  # [dim]
        
        # Predict raw architecture parameters
        arch_params = self.arch_predictor(task_embedding)
        
        # Scale parameters to appropriate ranges using sigmoid
        depth_param = torch.sigmoid(arch_params[0]) 
        heads_param = torch.sigmoid(arch_params[1])
        dim_head_param = torch.sigmoid(arch_params[2])
        mlp_param = torch.sigmoid(arch_params[3])
        
        # Convert to actual architecture parameters
        depth = int(torch.round(
            self.min_depth + (self.max_depth - self.min_depth) * depth_param
        ).item())
        
        heads = int(torch.round(
            self.min_heads + (self.max_heads - self.min_heads) * heads_param
        ).item())
        
        dim_head = int(torch.round(
            self.min_dim_head + (self.max_dim_head - self.min_dim_head) * dim_head_param
        ).item())
        
        mlp_dim = int(torch.round(
            self.min_mlp + (self.max_mlp - self.min_mlp) * mlp_param
        ).item())
        
        # Store raw parameters for gradient flow
        self.current_arch_params = {
            'depth_param': depth_param,
            'heads_param': heads_param,
            'dim_head_param': dim_head_param,
            'mlp_param': mlp_param
        }
        
        # Store actual architecture for monitoring
        arch_config = {
            'depth': depth,
            'heads': heads,
            'dim_head': dim_head,
            'mlp_dim': mlp_dim
        }
        
        if not self.training:
            self.arch_history.append(arch_config)
            
        return arch_config
    
    def set_forward(self, x, is_feature=False):
        # Extract features
        z_support, z_query = self.parse_feature(x, is_feature)
        
        # Reshape support features and compute prototypes
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
        
        # Reshape query features
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)
        
        # Predict architecture from support features
        arch_config = self.get_dynamic_architecture(z_support)
        depth = arch_config['depth']
        heads = arch_config['heads']
        dim_head = arch_config['dim_head']
        mlp_dim = arch_config['mlp_dim']
        
        x, query = z_proto, z_query
        
        # Apply transformer layers with dynamic architecture
        layer_weights = F.softmax(self.layer_importance[:depth], dim=0)
        
        for i in range(depth):
            # Apply layer with importance weighting
            layer_output = self.layers[i](
                q=x, k=query, v=query, 
                num_heads=heads, dim_head=dim_head, mlp_dim=mlp_dim
            )
            
            # Weighted residual connection
            x = x + layer_output * layer_weights[i]
        
        # Output projection with dynamic dimension
        return self.dynamic_linear(x, dim_head)
    
    def get_architecture_stats(self):
        """Returns statistics about the architecture configurations used"""
        if not self.arch_history:
            return None
        
        # Convert list of dicts to dict of lists
        stats = {k: [d[k] for d in self.arch_history] for k in self.arch_history[0]}
        
        # Compute statistics
        result = {}
        for k, v in stats.items():
            result[f'{k}_mean'] = float(np.mean(v))
            result[f'{k}_std'] = float(np.std(v))
            result[f'{k}_min'] = float(np.min(v))
            result[f'{k}_max'] = float(np.max(v))
            
        return result
    
    def clear_architecture_history(self):
        """Clear recorded architecture history"""
        self.arch_history = []


class DynamicTransformerLayer(nn.Module):
    """A transformer layer with dynamic architecture parameters"""
    def __init__(self, dim, max_heads, max_dim_head, max_mlp_dim, variant,
                initial_cov_weight, initial_var_weight, dynamic_weight):
        super().__init__()
        
        # Initialize with maximum dimensions
        max_inner_dim = max_heads * max_dim_head
        
        # Dynamic attention with maximum capacity
        self.attention = DynamicAttention(
            dim, max_heads, max_dim_head, variant,
            initial_cov_weight, initial_var_weight, dynamic_weight
        )
        
        # Dynamic layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dynamic MLP with maximum capacity
        self.mlp_up = nn.Linear(dim, max_mlp_dim)
        self.mlp_act = nn.GELU()
        self.mlp_down = nn.Linear(max_mlp_dim, dim)
        
    def forward(self, q, k, v, num_heads, dim_head, mlp_dim):
        # Apply attention with dynamic parameters
        attn_out = self.attention(
            q=self.norm1(q), 
            k=self.norm1(k), 
            v=self.norm1(v),
            num_heads=num_heads,
            dim_head=dim_head
        )
        
        # Apply first residual connection
        q = q + attn_out
        
        # Apply MLP with dynamic mlp_dim
        # Use only portion of the weights
        x = self.norm2(q)
        x = self.mlp_up(x)[:, :, :mlp_dim]
        x = self.mlp_act(x)
        x = self.mlp_down(x[:, :, :mlp_dim])
        
        # Apply second residual connection
        return q + x


class DynamicAttention(nn.Module):
    """Attention module with dynamic number of heads and dimension"""
    def __init__(self, dim, max_heads, max_dim_head, variant, 
                initial_cov_weight, initial_var_weight, dynamic_weight):
        super().__init__()
        max_inner_dim = max_heads * max_dim_head
        
        self.max_heads = max_heads
        self.variant = variant
        
        # Dynamic input projection
        self.input_linear = nn.Linear(dim, max_inner_dim, bias=False)
        
        # Dynamic output projection
        self.output_linear = nn.Linear(max_inner_dim, dim)
        
        # Dynamic weighting components (reuse from base class)
        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            # Network to predict weights based on features
            self.weight_predictor = nn.Sequential(
                nn.Linear(max_dim_head * 2, max_dim_head),
                nn.LayerNorm(max_dim_head),
                nn.ReLU(),
                nn.Linear(max_dim_head, 3),
                nn.Softmax(dim=-1)
            )
        else:
            # Fixed weights as parameters
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))
            
        # For weight recording
        self.weight_history = []
        self.record_weights = False
    
    def forward(self, q, k, v, num_heads, dim_head):
        # Compute inner dimension for current configuration
        inner_dim = num_heads * dim_head
        
        # Dynamic feature projection
        # Project and use only portion of the weights needed
        f_q = self.input_linear(q)[:, :, :inner_dim]
        f_k = self.input_linear(k)[:, :, :inner_dim]
        f_v = self.input_linear(v)[:, :, :inner_dim]
        
        # Reshape to multi-head attention
        f_q = rearrange(f_q, 'q n (h d) -> h q n d', h=num_heads)
        f_k = rearrange(f_k, 'q n (h d) -> h q n d', h=num_heads)
        f_v = rearrange(f_v, 'q n (h d) -> h q n d', h=num_heads)
        
        if self.variant == "cosine":
            # Calculate cosine similarity
            cosine_sim = cosine_distance(f_q, f_k.transpose(-1, -2))
            
            # Calculate covariance component
            q_centered = f_q - f_q.mean(dim=-1, keepdim=True)
            k_centered = f_k - f_k.mean(dim=-1, keepdim=True)
            cov_component = torch.matmul(q_centered, k_centered.transpose(-1, -2))
            cov_component = cov_component / f_q.size(-1)
            
            # Calculate variance component
            q_var = torch.var(f_q, dim=-1, keepdim=True)
            k_var = torch.var(f_k, dim=-1, keepdim=True).transpose(-1, -2)
            var_component = torch.matmul(q_var, k_var)
            var_component = var_component / f_q.size(-1)
            
            if self.dynamic_weight:
                # Use global feature statistics
                q_global = f_q.mean(dim=(1, 2))  # [h, d]
                k_global = f_k.mean(dim=(1, 2))  # [h, d]
                
                # Concatenate global query and key features, but only use dim_head dimensions
                qk_features = torch.cat([q_global[:, :dim_head], k_global[:, :dim_head]], dim=-1)
                
                # Predict weights
                weights = self.weight_predictor(qk_features)
                
                # Record weights during evaluation if needed
                if self.record_weights and not self.training:
                    self.weight_history.append(weights.detach().cpu().numpy().mean(axis=0))
                
                # Extract individual weights
                cos_weight = weights[:, 0].view(num_heads, 1, 1, 1)
                cov_weight = weights[:, 1].view(num_heads, 1, 1, 1)
                var_weight = weights[:, 2].view(num_heads, 1, 1, 1)
                
                # Combine all components
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
            else:
                # Use fixed weights
                cov_weight = torch.sigmoid(self.fixed_cov_weight)
                var_weight = torch.sigmoid(self.fixed_var_weight)
                cos_weight = 1.0 - cov_weight - var_weight
                
                dots = (cos_weight * cosine_sim + 
                       cov_weight * cov_component + 
                       var_weight * var_component)
                
            out = torch.matmul(dots, f_v)
        else:
            # Regular softmax attention
            scale = dim_head ** -0.5
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * scale
            dots = F.softmax(dots, dim=-1)
            out = torch.matmul(dots, f_v)
        
        # Reshape and project output
        out = rearrange(out, 'h q n d -> q n (h d)')
        
        # Use only the needed portion of the output projection
        return self.output_linear(out[:, :, :inner_dim])
    
    def get_weight_stats(self):
        """Returns statistics about the weights used"""
        if not self.weight_history:
            return None
        
        weights = np.array(self.weight_history)
        if weights.shape[1] == 3:
            return {
                'cosine_mean': float(weights[:, 0].mean()),
                'cov_mean': float(weights[:, 1].mean()),
                'var_mean': float(weights[:, 2].mean()),
                'cosine_std': float(weights[:, 0].std()),
                'cov_std': float(weights[:, 1].std()),
                'var_std': float(weights[:, 2].std())
            }
        else:
            return {
                'mean': float(weights.mean()),
                'std': float(weights.std()),
                'min': float(weights.min()),
                'max': float(weights.max())
            }
    
    def clear_weight_history(self):
        """Clear recorded weight history"""
        self.weight_history = []


class DynamicProjection(nn.Module):
    """Dynamic output projection layer that adapts to dimension size"""
    def __init__(self, dim, max_dim_head, variant):
        super().__init__()
        self.variant = variant
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
        
        # Linear projection to dimension
        self.linear = nn.Linear(dim, max_dim_head)
        
        # Final classification projection
        if variant == "cosine":
            self.classifier = CosineDistLinear(max_dim_head, 1)
        else:
            self.classifier = nn.Linear(max_dim_head, 1)
    
    def forward(self, x, dim_head):
        x = self.norm(x)
        x = self.linear(x)[:, :, :dim_head]  # Use only needed portion
        return self.classifier(x).squeeze()
