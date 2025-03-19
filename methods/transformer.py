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
                initial_cov_weight=0.2, initial_var_weight=0.2, dynamic_weight=False):
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
        
        classification_loss = self.loss_fn(scores, target)
        predict = torch.argmax(scores, dim = 1)
        acc = (predict == target).sum().item() / target.size(0)
        
        # Add VIC regularization
        support_features = z_support.reshape(-1, z_support.size(-1))
        
        # Variance regularization
        std_loss = torch.mean(F.relu(0.5 - torch.sqrt(torch.var(support_features, dim=0) + 1e-5)))
        
        # Covariance regularization
        z_centered = support_features - support_features.mean(0)
        cov = (z_centered.T @ z_centered) / (z_centered.size(0) - 1)
        cov_reg = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / support_features.size(1)
        
        # Combined loss
        loss = classification_loss + 0.1 * std_loss + 0.01 * cov_reg
        
        return acc, loss

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.6, initial_var_weight=0.2, dynamic_weight=False):
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
            
            # Enhanced covariance component for better decorrelation
            # Center features properly
            q_centered = f_q - f_q.mean(dim=-1, keepdim=True)  # [h, q, n, d]
            k_centered = f_k - f_k.mean(dim=-1, keepdim=True)  # [h, q, m, d]
            
            # Compute feature dimension for normalization
            d = f_q.size(-1)
            
            # For each head, query position, and query/key pair:
            # Create improved covariance component that promotes decorrelation
            
            # Method 1: Full decorrelation using off-diagonal covariance
            # Reshape to process pairs efficiently
            batch_size = f_q.size(1) * f_q.size(2)  # q*n
            key_size = f_k.size(2)  # m
            
            # Process each head separately to reduce memory
            cov_matrices = []
            for h in range(f_q.shape[0]):  # For each head
                # Compute per-head covariance values
                head_cov_values = []
                
                for i in range(f_q.size(1)):  # For each query position
                    q_feat = q_centered[h, i]  # [n, d]
                    k_feat = k_centered[h, i]  # [m, d]
                    
                    # Joint features for this query position
                    z_joint = torch.cat([q_feat, k_feat], dim=0)  # [n+m, d]
                    
                    # Compute covariance matrix
                    cov_matrix = torch.matmul(z_joint.transpose(-2, -1), z_joint) / (q_feat.size(0) + k_feat.size(0) - 1)  # [d, d]
                    
                    # Isolate off-diagonal elements (the correlations)
                    off_diag = cov_matrix - torch.diag(torch.diagonal(cov_matrix))  # [d, d]
                    
                    # Calculate Frobenius norm of off-diagonal elements (squared and summed)
                    off_diag_norm = torch.sum(off_diag.pow(2)) / d
                    
                    # Create attention values that are higher when decorrelation is better (lower off-diag)
                    decorr_factor = 1.0 / (1.0 + off_diag_norm)  # Inversely proportional to correlation
                    
                    # Expand to create attention values for each n,m pair
                    att_values = decorr_factor * torch.ones(q_feat.size(0), k_feat.size(0), device=q_feat.device)
                    head_cov_values.append(att_values)
                
                # Stack for this head
                head_matrix = torch.stack(head_cov_values, dim=0)  # [q, n, m]
                cov_matrices.append(head_matrix)
            
            # Stack across heads
            cov_component = torch.stack(cov_matrices, dim=0)  # [h, q, n, m]
            
            # Normalize to similar scale as other components
            cov_component = cov_component / torch.max(cov_component)
            
            # Continue with rest of attention mechanism (variance and weighting)
            # Improved variance component using standard deviation
            # Compute per-feature variance, not just total variance
            q_var = torch.var(f_q, dim=-1, unbiased=True)  # [h, q, n]
            k_var = torch.var(f_k, dim=-1, unbiased=True)  # [h, q, m]

            # Target variance (encourage feature diversity)
            target_var = 1.0

            # Create variance ratio that peaks at optimal variance
            q_var_ratio = 2 * torch.min(q_var, target_var) / (q_var + target_var)
            k_var_ratio = 2 * torch.min(k_var, target_var) / (k_var + target_var)

            # Create variance component through outer product
            var_component = torch.bmm(
                q_var_ratio.view(f_q.shape[0], -1), 
                k_var_ratio.view(f_k.shape[0], -1).transpose(-2, -1)
            ).view_as(cosine_sim)
            
            # More efficient covariance calculation
            q_flat = q_centered.reshape(f_q.shape[0], -1, f_q.shape[-1])  # [h, q*n, d]
            k_flat = k_centered.reshape(f_k.shape[0], -1, f_k.shape[-1])  # [h, q*m, d]

            # Calculate feature correlation matrix
            feature_corr = torch.bmm(q_flat.transpose(-2, -1), q_flat) / q_flat.size(1)  # [h, d, d]
            feature_corr = feature_corr - torch.diag_embed(torch.diagonal(feature_corr, dim1=-2, dim2=-1))

            # Compute decorrelation score (lower means better decorrelated)
            decorr_score = torch.norm(feature_corr, p='fro', dim=(-2, -1)) / (f_q.size(-1) * f_q.size(-1))
            decorr_scale = torch.exp(-5.0 * decorr_score).unsqueeze(-1).unsqueeze(-1)  # [h, 1, 1]

            # Apply as scaling factor to cosine similarity
            cov_component = cosine_sim * decorr_scale
            
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
