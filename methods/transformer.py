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
from torch.cuda.amp import autocast

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
        
        # Replace static proto_weight with dynamic weight network
        self.weight_network = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head),
            nn.GELU(),
            nn.Linear(dim_head, 1)
        )
        
        # Compatibility projections for query-support matching
        self.query_proj = nn.Linear(dim, dim)
        self.support_proj = nn.Linear(dim, dim)
        
        # Cache for feature reuse
        self.feature_cache = {}
        
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
        
    def compute_dynamic_weights(self, support, query):
        """Compute weights based on query-support compatibility"""
        batch_size = query.shape[0]
        
        # Check cache first to avoid recomputation
        cache_key = f"query_{batch_size}"
        if (cache_key in self.feature_cache):
            q_proj = self.feature_cache[cache_key]
        else:
            # Project query features - do once for efficiency
            with autocast(enabled=torch.cuda.is_available()):
                q_proj = self.query_proj(query)
                self.feature_cache[cache_key] = q_proj
        
        # Project support features
        with autocast(enabled=torch.cuda.is_available()):
            s_proj = self.support_proj(support)  # [n_way, k_shot, dim]
            
            # Calculate average query embedding per class to reduce computation
            q_avg = q_proj.view(self.n_way, self.n_query, -1).mean(1)  # [n_way, dim]
            q_expanded = q_avg.unsqueeze(1).expand(-1, self.k_shot, -1)  # [n_way, k_shot, dim]
            
            # Get input for weight network by combining support and query information
            # Use element-wise addition for faster computation than concatenation
            weight_input = s_proj + q_expanded
            
            # Generate weights through the network
            attention_logits = self.weight_network(weight_input)  # [n_way, k_shot, 1]
            
            # Apply softmax per class for proper weighting
            attention_weights = F.softmax(attention_logits, dim=1)  # [n_way, k_shot, 1]
            
        return attention_weights
        
    def set_forward(self, x, is_feature=False):
        # First try to offload unused parameters to CPU
        if hasattr(self, 'offload_unused_modules') and not self.training:
            # Only keep active modules on GPU during inference
            self.feature = self.feature.to(device)
            self.ATTN = self.ATTN.to('cpu')
            torch.cuda.empty_cache()
        
        # Extract features with optimized memory
        z_support, z_query = self.parse_feature(x, is_feature)
        
        # Now bring attention modules back to GPU
        self.ATTN = self.ATTN.to(device)
        torch.cuda.empty_cache()
        
        # Use half-precision for feature extraction where possible
        with torch.cuda.amp.autocast():
            z_support, z_query = self.parse_feature(x, is_feature)
            
            # Move to CPU if needed to save GPU memory, then back when needed
            if not self.training:
                torch.cuda.empty_cache()  # Clear GPU memory
                
            z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
            z_query_flat = z_query.contiguous().view(self.n_way * self.n_query, -1)
            
            # Compute dynamic weights with memory efficiency
            dynamic_weights = self.compute_dynamic_weights(z_support, z_query_flat)
            
            # Generate prototypes with dynamic weights
            z_proto = (z_support * dynamic_weights).sum(1, keepdim=True)  # [n_way, 1, dim]
            
            # Free up memory
            del z_support
            
            # Process in smaller chunks if query set is large
            chunk_size = 10  # Adjust based on memory
            scores_list = []
            
            for i in range(0, self.n_way * self.n_query, chunk_size):
                query_chunk = z_query_flat[i:i+chunk_size].unsqueeze(1)  # [chunk, 1, dim]
                
                # Process through transformer for this chunk
                x = z_proto.transpose(0, 1)  # [1, n_way, dim]
                query = query_chunk  # [chunk, 1, dim]
                
                for _ in range(self.depth):
                    x = x + self.ATTN(q=x, k=query, v=query)
                    x = x + self.FFN(x)
                
                # Get scores for this chunk
                chunk_scores = self.linear(x).squeeze()  # [chunk, n_way]
                scores_list.append(chunk_scores)
                
                # Clear memory after each chunk
                del query_chunk, x, query
                torch.cuda.empty_cache()
            
            # Combine results from all chunks
            scores = torch.cat(scores_list, dim=0)
            return scores
    
    def set_forward_loss(self, x):
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))  # this is the target groundtruth
        scores = self.set_forward(x)
        
        loss = self.loss_fn(scores, target)
        predict = torch.argmax(scores, dim = 1)
        acc = (predict == target).sum().item() / target.size(0)
        return acc, loss

    def clear_cache(self):
        self.feature_cache = {}
        if hasattr(self, '_feature_cache'):
            self._feature_cache = {}

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

