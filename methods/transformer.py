import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from methods.meta_template import MetaTemplate
from methods.vic_regularization import VICRegularization, DynamicVICWeights
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from backbone import CosineDistLinear
import pdb
import IPython

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func,  n_way, k_shot, n_query, variant = "softmax",
                depth = 1, heads = 8, dim_head = 64, mlp_dim = 512,
                use_vic=False, lambda_V_base=0.5, lambda_I=9.0, lambda_C_base=0.5,
                vic_gamma=1.0, vic_epsilon=1e-6):
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
        
        # VIC Regularization components
        self.use_vic = use_vic
        if self.use_vic:
            self.vic_reg = VICRegularization(gamma=vic_gamma, epsilon=vic_epsilon)
            self.vic_weights = DynamicVICWeights(
                lambda_V_base=lambda_V_base,
                lambda_I=lambda_I,
                lambda_C_base=lambda_C_base
            )
        
        # Store embeddings for VIC regularization
        self.z_support_cache = None
        self.z_proto_cache = None
        
    def set_forward(self, x, is_feature=False):

        z_support, z_query = self.parse_feature(x, is_feature)
                
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)                         # (1, n, d)
        
        # Cache embeddings for VIC regularization if enabled
        if self.use_vic and self.training:
            self.z_support_cache = z_support
            self.z_proto_cache = z_proto.squeeze(0)  # (n, d)
        
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)                # (q, 1, d)

        x, query = z_proto, z_query
        
        for _ in range(self.depth):
           x = self.ATTN(q = x, k = query, v = query) + x
           x = self.FFN(x) + x
        
        # Output is the probabilistic prediction for each class
        return self.linear(x).squeeze()                                                                # (q, n)
    
    def set_forward_loss(self, x, current_epoch=0, total_epochs=50):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        scores = self.set_forward(x)
        
        # Invariance loss (classification loss)
        loss_ce = self.loss_fn(scores, target)
        
        # Add VIC regularization if enabled
        if self.use_vic and self.training and self.z_support_cache is not None:
            # Get dynamic weights
            weights = self.vic_weights.get_weights(current_epoch, total_epochs)
            
            # Concatenate support embeddings and prototypes
            # z_support_cache: (n, k, d), z_proto_cache: (n, d)
            z_support_flat = self.z_support_cache.view(-1, self.z_support_cache.size(-1))  # (n*k, d)
            embeddings = torch.cat([z_support_flat, self.z_proto_cache], dim=0)  # (n*k + n, d)
            
            # Compute VIC losses
            vic_losses = self.vic_reg(embeddings)
            
            # Combined loss with dynamic weights
            loss = (weights['lambda_I'] * loss_ce + 
                   weights['lambda_V'] * vic_losses['variance_loss'] +
                   weights['lambda_C'] * vic_losses['covariance_loss'])
            
            # Normalize by invariance weight to keep loss scale similar
            loss = loss / weights['lambda_I']
        else:
            loss = loss_ce
        
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
