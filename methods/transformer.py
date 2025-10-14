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
                gamma = 0.1, epsilon = 1e-8, use_regularization = True):
        super(FewShotTransformer, self).__init__(model_func,  n_way, k_shot, n_query)

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        dim = self.feat_dim

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
        
        # Regularization parameters
        self.gamma = gamma  # Target value for variance (fixed to 0.1 in experiments)
        self.epsilon = epsilon  # Small scalar preventing numerical instability
        self.use_regularization = use_regularization
        
        # Dynamic weight predictor for combining three loss components
        if use_regularization:
            # Input: global statistics from embeddings
            # Output: 3 weights (for cross-entropy, variance, covariance)
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim * 2, dim),  # Concatenate support and query embeddings
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, 3),
                nn.Softmax(dim=-1)  # Ensure weights sum to 1
            )
        
    def variance_regularization(self, E):
        """
        Compute Variance Regularization Term (Equation 5)
        V(E) = (1/m) * sum_j max(0, gamma - sigma(E_j, epsilon))
        where sigma(E_j, epsilon) = sqrt(Var(E_j) + epsilon)
        
        Args:
            E: Embeddings tensor of shape (batch, m) where m is number of dimensions
        Returns:
            V: Variance regularization term (scalar)
        """
        # Compute variance for each dimension
        var_per_dim = torch.var(E, dim=0)  # (m,)
        
        # Compute regularized standard deviation
        sigma = torch.sqrt(var_per_dim + self.epsilon)  # (m,)
        
        # Compute hinge function
        V = torch.mean(torch.clamp(self.gamma - sigma, min=0.0))
        
        return V
    
    def covariance_regularization(self, E):
        """
        Compute Covariance Regularization Term (Equation 6)
        C(E) = (1/(m-1)) * sum_j (E_j - E_bar)(E_j - E_bar)^T
        where E_bar = (1/K) * sum_i E_j
        
        This computes the sum of squared off-diagonal coefficients of the covariance matrix.
        
        Args:
            E: Embeddings tensor of shape (batch, m) where m is number of dimensions
        Returns:
            C: Covariance regularization term (scalar)
        """
        # Center the embeddings
        E_mean = torch.mean(E, dim=0, keepdim=True)  # (1, m)
        E_centered = E - E_mean  # (batch, m)
        
        # Compute covariance matrix
        batch_size = E.size(0)
        if batch_size > 1:
            cov = torch.matmul(E_centered.T, E_centered) / (batch_size - 1)  # (m, m)
        else:
            cov = torch.matmul(E_centered.T, E_centered)  # (m, m)
        
        # Sum of squares of off-diagonal elements
        m = E.size(1)
        off_diag_mask = ~torch.eye(m, dtype=torch.bool, device=E.device)
        C = torch.sum(cov[off_diag_mask] ** 2) / m
        
        return C
    
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
        
        # Get embeddings for regularization computation
        z_support, z_query = self.parse_feature(x, is_feature=False)
        
        # Get prediction scores
        scores = self.set_forward(x)
        
        # Compute cross-entropy loss (Equation 4)
        ce_loss = self.loss_fn(scores, target)
        
        if self.use_regularization:
            # Concatenate support and query embeddings for regularization
            # E is the concatenation of support set embedding E_k and prototype embedding P_k
            z_support_flat = z_support.contiguous().view(-1, z_support.size(-1))  # (n_way * k_shot, d)
            z_query_flat = z_query.contiguous().view(-1, z_query.size(-1))  # (n_way * n_query, d)
            E = torch.cat([z_support_flat, z_query_flat], dim=0)  # (n_way * (k_shot + n_query), d)
            
            # Compute variance regularization (Equation 5)
            var_reg = self.variance_regularization(E)
            
            # Compute covariance regularization (Equation 6)
            cov_reg = self.covariance_regularization(E)
            
            # Predict dynamic weights using global statistics
            support_global = z_support_flat.mean(dim=0)  # (d,)
            query_global = z_query_flat.mean(dim=0)  # (d,)
            global_features = torch.cat([support_global, query_global], dim=-1).unsqueeze(0)  # (1, 2*d)
            
            weights = self.weight_predictor(global_features).squeeze(0)  # (3,)
            w_ce, w_var, w_cov = weights[0], weights[1], weights[2]
            
            # Combine losses with dynamic weights
            loss = w_ce * ce_loss + w_var * var_reg + w_cov * cov_reg
        else:
            loss = ce_loss
        
        predict = torch.argmax(scores, dim = 1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, loss

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
