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
    def __init__(self, model_func,  n_way, k_shot, n_query, variant = "softmax",
                depth = 1, heads = 8, dim_head = 64, mlp_dim = 512,
                lambda_I = 1.0, lambda_V = 0.0, lambda_C = 0.0):
        super(FewShotTransformer, self).__init__(model_func,  n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim
        
        # VIC loss weights
        self.lambda_I = lambda_I
        self.lambda_V = lambda_V
        self.lambda_C = lambda_C

        self.ATTN = Attention(dim, heads = heads, dim_head = dim_head, variant = variant)
        
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
    
    def variance_loss(self, z_support):
        """
        Variance Loss (L_V): Hinge loss on standard deviation of support embeddings
        Encourages compactness of support set embeddings for each class.
        
        Args:
            z_support: Support set embeddings, shape (n_way, k_shot, feat_dim)
        
        Returns:
            Variance loss value
        """
        # Calculate standard deviation for each class across the support samples
        # z_support shape: (n_way, k_shot, feat_dim)
        std_per_class = torch.std(z_support, dim=1)  # Shape: (n_way, feat_dim)
        
        # Apply hinge loss: max(0, gamma - std) to encourage std < gamma
        # Using gamma = 1.0 as a typical threshold
        gamma = 1.0
        hinge_loss = F.relu(gamma - std_per_class)
        
        # Average over all classes and feature dimensions
        loss_v = hinge_loss.mean()
        
        return loss_v
    
    def covariance_loss(self, z_support):
        """
        Covariance Loss (L_C): Covariance regularization to decorrelate feature dimensions
        Prevents informational collapse by encouraging feature dimensions to be independent.
        
        Args:
            z_support: Support set embeddings, shape (n_way, k_shot, feat_dim)
        
        Returns:
            Covariance loss value
        """
        # Flatten support embeddings across all classes and samples
        # z_support shape: (n_way, k_shot, feat_dim)
        batch_size = z_support.size(0) * z_support.size(1)
        feat_dim = z_support.size(2)
        z_flat = z_support.view(batch_size, feat_dim)  # Shape: (n_way*k_shot, feat_dim)
        
        # Center the features (subtract mean)
        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov_matrix = (z_centered.T @ z_centered) / (batch_size - 1)  # Shape: (feat_dim, feat_dim)
        
        # Covariance loss: sum of squared off-diagonal elements
        # We want to minimize correlation between different feature dimensions
        loss_c = (cov_matrix ** 2).sum() - (torch.diag(cov_matrix) ** 2).sum()
        loss_c = loss_c / (feat_dim * (feat_dim - 1))  # Normalize by number of off-diagonal elements
        
        return loss_c
    
    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        
        # Extract support embeddings for VIC losses
        z_support, z_query = self.parse_feature(x, is_feature=False)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        
        # Get prediction scores
        scores = self.set_forward(x)
        
        # Invariance Loss (L_I): Standard Cross-Entropy Loss
        loss_I = self.loss_fn(scores, target)
        
        # Variance Loss (L_V): Hinge loss on support set standard deviation
        loss_V = self.variance_loss(z_support) if self.lambda_V > 0 else torch.tensor(0.0, device=device)
        
        # Covariance Loss (L_C): Covariance regularization
        loss_C = self.covariance_loss(z_support) if self.lambda_C > 0 else torch.tensor(0.0, device=device)
        
        # Combined Loss: L_total = (λ_I * L_I) + (λ_V * L_V) + (λ_C * L_C)
        loss_total = (self.lambda_I * loss_I) + (self.lambda_V * loss_V) + (self.lambda_C * loss_C)
        
        predict = torch.argmax(scores, dim = 1)
        acc = (predict == target).sum().item() / target.size(0)
        
        return acc, loss_total

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim = -1)
        self.variant = variant
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias = False))
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        
    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) ->  h q n d', h = self.heads), (q, k ,v))    
        
        if self.variant == "cosine":
            dots = cosine_distance(f_q, f_k.transpose(-1, -2))                                         # (h, q, n, 1)
            out = torch.matmul(dots, f_v)                                                              # (h, q, n, d_h)
        
        else: # self.variant == "softmax"
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale            
            out = torch.matmul(self.sm(dots), f_v)
        
        out = rearrange(out, 'h q n d -> q n (h d)')                                                   # (q, n, d)
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
