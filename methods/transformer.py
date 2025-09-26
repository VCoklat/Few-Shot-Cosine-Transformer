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
import wandb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ProgressiveRegularizationScheduler:
    """
    Implements progressive regularization schedule with variance and covariance terms
    from the mathematical formulations in the paper.
    """
    
    def __init__(self, initial_dropout=0.02, initial_weight_decay=1e-6, 
                 max_dropout=0.25, max_weight_decay=5e-4,
                 warmup_epochs=15, total_epochs=100,
                 variance_weight=0.1, covariance_weight=0.05):
        
        self.initial_dropout = initial_dropout
        self.initial_weight_decay = initial_weight_decay
        self.max_dropout = max_dropout
        self.max_weight_decay = max_weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        
    def get_dropout_rate(self, epoch):
        """Calculate dropout rate for current epoch"""
        if epoch < self.warmup_epochs:
            return self.initial_dropout
        else:
            progress = min(1.0, (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))
            return self.initial_dropout + (self.max_dropout - self.initial_dropout) * progress
    
    def get_weight_decay(self, epoch):
        """Calculate weight decay for current epoch"""
        if epoch < self.warmup_epochs:
            return self.initial_weight_decay
        else:
            progress = min(1.0, (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))
            return self.initial_weight_decay + (self.max_weight_decay - self.initial_weight_decay) * progress
    
    def variance_regularization(self, embeddings):
        """
        Implement variance regularization V(E) from equation (5)
        V(E) = 1/m * sum(max(0, γ - σ(Ej, ej)))
        """
        if embeddings.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        m = embeddings.size(0)
        regularization_loss = 0
        
        # For each embedding in the batch
        for i in range(m):
            # Calculate standard deviation along feature dimension
            std_dev = torch.std(embeddings[i])
            # Apply max(0, γ - std) where γ is regularization target (fixed to 1)
            regularization_loss += torch.clamp(1.0 - std_dev, min=0)
        
        return (self.variance_weight / m) * regularization_loss
    
    def covariance_regularization(self, embeddings):
        """
        Implement covariance regularization C(E) from equation (6)
        C(E) = 1/(m-1) * sum((Ej - E̅)(Ej - E̅)^T) with K/m-1 scaling factor
        """
        m = embeddings.size(0)
        if m <= 1:
            return torch.tensor(0.0, device=embeddings.device)
            
        # Center the embeddings (subtract mean)
        mean_embedding = torch.mean(embeddings, dim=0, keepdim=True)
        centered_embeddings = embeddings - mean_embedding
        
        # Calculate empirical covariance matrix with K/(m-1) factor
        K = embeddings.size(-1)  # Feature dimension
        covariance_matrix = torch.matmul(centered_embeddings.t(), centered_embeddings) * (K / (m - 1))
        
        # Extract off-diagonal elements (encourage decorrelation)
        mask = torch.eye(covariance_matrix.size(0), device=embeddings.device)
        off_diagonal = covariance_matrix * (1 - mask)
        
        # Sum of squared off-diagonal coefficients
        covariance_loss = torch.sum(off_diagonal ** 2)
        
        return self.covariance_weight * covariance_loss


class FewShotTransformer(MetaTemplate):
    def __init__(self, model_func, n_way, k_shot, n_query, variant="softmax",
                 depth=1, heads=8, dim_head=64, mlp_dim=512,
                 initial_cov_weight=0.3, initial_var_weight=0.5, dynamic_weight=False,
                 enable_progressive_reg=True, reg_scheduler_params=None):
        
        super(FewShotTransformer, self).__init__(model_func, n_way, k_shot, n_query)
        self.loss_fn = nn.CrossEntropyLoss()
        self.k_shot = k_shot
        self.variant = variant
        self.depth = depth
        self.enable_progressive_reg = enable_progressive_reg
        
        dim = self.feat_dim
        
        # Initialize attention module
        self.ATTN = Attention(dim, heads=heads, dim_head=dim_head, variant=variant,
                             initial_cov_weight=initial_cov_weight,
                             initial_var_weight=initial_var_weight,
                             dynamic_weight=dynamic_weight)
        
        self.sm = nn.Softmax(dim=-2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        
        # Enhanced FFN with progressive dropout
        self.FFN = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=0.02),  # Start with minimal dropout
            nn.Linear(mlp_dim, dim),
            nn.Dropout(p=0.02)   # Progressive dropout
        )
        
        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head),
            CosineDistLinear(dim_head, 1) if variant == "cosine"
            else nn.Linear(dim_head, 1)
        )
        
        # Progressive regularization components
        if self.enable_progressive_reg:
            if reg_scheduler_params is None:
                reg_scheduler_params = {
                    'initial_dropout': 0.02,
                    'max_dropout': 0.25,
                    'initial_weight_decay': 1e-6,
                    'max_weight_decay': 5e-4,
                    'warmup_epochs': 15,
                    'total_epochs': 100,
                    'variance_weight': 0.1,
                    'covariance_weight': 0.05
                }
            
            self.reg_scheduler = ProgressiveRegularizationScheduler(**reg_scheduler_params)
            self.current_epoch = 0
            self.adaptive_dropout = nn.Dropout(p=0.02)
        
    def update_regularization(self, epoch):
        """Update regularization parameters for current epoch"""
        if not self.enable_progressive_reg:
            return
            
        self.current_epoch = epoch
        dropout_rate = self.reg_scheduler.get_dropout_rate(epoch)
        
        # Update dropout in FFN
        for module in self.FFN.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
        
        # Update adaptive dropout
        self.adaptive_dropout.p = dropout_rate
        
        print(f"Epoch {epoch}: Dropout={dropout_rate:.4f}, "
              f"Weight_decay={self.reg_scheduler.get_weight_decay(epoch):.6f}")
    
    def get_current_weight_decay(self):
        """Get current weight decay for optimizer"""
        if not self.enable_progressive_reg:
            return 0.0
        return self.reg_scheduler.get_weight_decay(self.current_epoch)
    
    def set_forward(self, x, is_feature=False):
        """Standard forward pass"""
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)
        
        x, query = z_proto, z_query
        
        for _ in range(self.depth):
            # Apply attention with residual connection
            attn_out = self.ATTN(q=x, k=query, v=query)
            
            # Apply adaptive dropout during training
            if self.training and self.enable_progressive_reg and self.current_epoch >= 5:
                attn_out = self.adaptive_dropout(attn_out)
            
            x = attn_out + x
            x = self.FFN(x) + x
        
        # Output is the probabilistic prediction for each class
        return self.linear(x).squeeze()  # (q, n)
    
    def set_forward_with_regularization(self, x, is_feature=False):
        """Forward pass with regularization loss computation"""
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_proto = (z_support * self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # (1, n, d)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)  # (q, 1, d)
        
        x, query = z_proto, z_query
        
        # Store embeddings for regularization
        embeddings_for_reg = []
        
        for depth_idx in range(self.depth):
            # Apply attention with residual connection
            attn_out = self.ATTN(q=x, k=query, v=query)
            
            # Apply adaptive dropout during training
            if self.training and self.enable_progressive_reg and self.current_epoch >= 5:
                attn_out = self.adaptive_dropout(attn_out)
            
            x = attn_out + x
            
            # Store embeddings for regularization (after attention but before FFN)
            if self.training and self.enable_progressive_reg:
                embeddings_for_reg.append(x.view(-1, x.size(-1)))
            
            x = self.FFN(x) + x
        
        # Compute regularization loss
        reg_loss = torch.tensor(0.0, device=x.device)
        if (self.training and self.enable_progressive_reg and 
            self.current_epoch >= self.reg_scheduler.warmup_epochs and embeddings_for_reg):
            
            # Use embeddings from the final layer for regularization
            final_embeddings = embeddings_for_reg[-1]
            
            # Apply variance and covariance regularization
            var_loss = self.reg_scheduler.variance_regularization(final_embeddings)
            cov_loss = self.reg_scheduler.covariance_regularization(final_embeddings)
            reg_loss = var_loss + cov_loss
        
        # Output is the probabilistic prediction for each class
        output = self.linear(x).squeeze()  # (q, n)
        return output, reg_loss
    
    def set_forward_loss(self, x):
        """Standard loss computation"""
        target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        target = Variable(target.to(device))
        
        if self.enable_progressive_reg:
            scores, reg_loss = self.set_forward_with_regularization(x)
            classification_loss = self.loss_fn(scores, target)
            total_loss = classification_loss + reg_loss
        else:
            scores = self.set_forward(x)
            total_loss = self.loss_fn(scores, target)
            reg_loss = torch.tensor(0.0)
        
        predict = torch.argmax(scores, dim=1)
        acc = (predict == target).sum().item() / target.size(0)
        
        return acc, total_loss, reg_loss
    
    def train_loop_with_regularization(self, epoch, num_epoch, base_loader, wandb_log, optimizer):
        """Enhanced training loop with progressive regularization"""
        print_freq = 10
        avg_loss = 0
        avg_reg_loss = 0
        avg_class_loss = 0
        
        # Update regularization for current epoch
        self.update_regularization(epoch)
        
        for i, (x, _) in enumerate(base_loader):
            self.train()
            optimizer.zero_grad()
            
            # Forward pass with regularization
            scores, reg_loss = self.set_forward_with_regularization(x)
            
            # Standard classification loss
            target = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
            target = Variable(target.to(device))
            classification_loss = self.loss_fn(scores, target)
            
            # Total loss combines classification and regularization
            total_loss = classification_loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            avg_loss += total_loss.item()
            avg_reg_loss += reg_loss.item()
            avg_class_loss += classification_loss.item()
            
            if i % print_freq == 0:
                print(f'Epoch {epoch:d} | Batch {i:d}/{len(base_loader):d} | '
                      f'Total Loss {avg_loss/float(i+1):.6f} | '
                      f'Class Loss {avg_class_loss/float(i+1):.6f} | '
                      f'Reg Loss {avg_reg_loss/float(i+1):.6f}')
        
        if wandb_log:
            wandb.log({
                'train_total_loss': avg_loss/len(base_loader),
                'train_class_loss': avg_class_loss/len(base_loader),
                'train_reg_loss': avg_reg_loss/len(base_loader),
                'dropout_rate': self.reg_scheduler.get_dropout_rate(epoch) if self.enable_progressive_reg else 0.0,
                'weight_decay': self.reg_scheduler.get_weight_decay(epoch) if self.enable_progressive_reg else 0.0,
                'epoch': epoch
            })


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, variant, initial_cov_weight=0.6, initial_var_weight=0.2, dynamic_weight=False):
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
            # Network to predict the weights based on features (now 3 components)
            self.weight_predictor = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.LayerNorm(dim_head),
                nn.ReLU(),
                nn.Dropout(p=0.1),  # Add dropout for stability
                nn.Linear(dim_head, 3),  # Now predict 3 weights instead of 1
                nn.Softmax(dim=-1)  # Ensure weights sum to 1.0
            )
        else:
            # Fixed weights as parameters (still learnable)
            self.fixed_cov_weight = nn.Parameter(torch.tensor(initial_cov_weight))
            self.fixed_var_weight = nn.Parameter(torch.tensor(initial_var_weight))
        
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False)
        )
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        
        self.weight_history = []  # To store weights for analysis
        self.record_weights = False  # Toggle for weight recording
        
    def forward(self, q, k, v):
        f_q, f_k, f_v = map(lambda t: rearrange(
            self.input_linear(t), 'q n (h d) -> h q n d', h=self.heads), (q, k, v))
        
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
                cos_weight = torch.clamp(1.0 - cov_weight - var_weight, min=0.1, max=0.8)
                
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
    x1 = [b, h, n, k]
    x2 = [b, h, k, m]
    output = [b, h, n, m]
    '''
    dots = torch.matmul(x1, x2)
    scale = torch.einsum('bhi, bhj -> bhij',
                        (torch.norm(x1, 2, dim=-1), torch.norm(x2, 2, dim=-2)))
    # Add small epsilon to prevent division by zero
    scale = torch.clamp(scale, min=1e-8)
    return (dots / scale)
