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
                depth = 1, heads = 8, dim_head = 64, mlp_dim = 512):
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
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.3, dynamic_weight=True):
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
            # Network to predict the weight based on features
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.LayerNorm(dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, 1),
                nn.Sigmoid()  # Output between 0-1
            )
        else:
            # Fixed weight as parameter (still learnable)
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias = False))
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        
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
            
            # Determine weight dynamically
            if self.dynamic_weight:
                # Create feature representations for weight prediction
                # Use average pooling over spatial dimensions
                q_pool = f_q.mean(dim=2)  # [h, q, d]
                k_pool = f_k.mean(dim=2)  # [h, q, d]
                
                # Concatenate query and key features
                qk_features = torch.cat([q_pool, k_pool], dim=-1)  # [h, q, 2d]
                
                # Reshape for weight prediction
                qk_flat = qk_features.reshape(-1, qk_features.size(-1))  # [h*q, 2d]
                
                # Predict weight
                weights = self.weight_predictor(qk_flat)  # [h*q, 1]
                weights = weights.reshape(f_q.size(0), f_q.size(1), 1, 1)  # [h, q, 1, 1]
                
                # Apply weight
                dots = (1 - weights) * cosine_sim + weights * cov_component
            else:
                # Use fixed but learnable weight
                cov_weight = torch.sigmoid(self.fixed_cov_weight)  # Constrain between 0-1
                dots = (1 - cov_weight) * cosine_sim + cov_weight * cov_component
                
            out = torch.matmul(dots, f_v)
        
        else: # self.variant == "softmax"
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
