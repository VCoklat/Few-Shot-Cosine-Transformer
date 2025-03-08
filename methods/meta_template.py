import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import tqdm
from abc import abstractmethod
import pdb
import wandb
import torch.cuda.amp as amp
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore")
class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, k_shot, n_query, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.k_shot     = k_shot
        self.n_query    = n_query
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature, cache_support=False):
        x = Variable(x.to(device))
        if is_feature:
            z_all = x
        else:
            # Check if we've already computed features for this input
            if hasattr(self, '_feature_cache') and self._feature_cache.get('x_id') == id(x) and not self.training:
                z_all = self._feature_cache.get('features')
            else:
                # Process in chunks to save memory
                chunk_size = 5  # Use smaller chunk size
                
                # Properly reshape: flatten the first two dimensions to batch dimension
                shots_per_class = x.size(1)  # This is k_shot + n_query
                x_flat = x.reshape(-1, *x.shape[2:])  # [n_way*(k_shot+n_query), C, H, W]
                
                # Create chunks of the flattened tensor
                total_samples = x_flat.size(0)
                num_chunks = max(1, total_samples // chunk_size)
                x_chunks = torch.chunk(x_flat, num_chunks, dim=0)
                
                # Process each chunk separately
                features = []
                for x_chunk in x_chunks:
                    with torch.cuda.amp.autocast(enabled=True):
                        feat_chunk = self.feature.forward(x_chunk)
                    features.append(feat_chunk)
                    torch.cuda.empty_cache()
                
                # Concatenate features and reshape back
                z_all_flat = torch.cat(features, dim=0)
                
                # Reshape back to [n_way, shots_per_class, feature_dim]
                z_all = z_all_flat.view(self.n_way, shots_per_class, *z_all_flat.shape[1:])
                
                # Cache features during evaluation
                if not self.training:
                    if not hasattr(self, '_feature_cache'):
                        self._feature_cache = {}
                    self._feature_cache['x_id'] = id(x)
                    self._feature_cache['features'] = z_all
                    
        z_support = z_all[:, :self.k_shot]
        z_query = z_all[:, self.k_shot:]

        return z_support, z_query

    def correct(self, x):       
        scores = self.set_forward(x)
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.to(device))

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels
        top1_correct = (topk_ind[:,0] == y_query).sum().item()
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, num_epoch, train_loader, wandb_flag, optimizer, use_amp=True, accumulation_steps=8):
        avg_loss = 0
        avg_acc = []
        
        # Force using mixed precision
        scaler = amp.GradScaler()
        
        # Clear cache
        torch.cuda.empty_cache()
        
        with tqdm.tqdm(total = len(train_loader)) as train_pbar:
            for i, (x, _) in enumerate(train_loader):
                if self.change_way:
                    self.n_way = x.size(0)
                
                # Zero gradients at appropriate intervals
                if i % accumulation_steps == 0:
                    optimizer.zero_grad()
                
                # Try gradual loading to GPU to avoid spikes
                if isinstance(x, list):
                    x = [item.to(device) for item in x]
                else:
                    x = x.to(device)
                
                # Always use mixed precision
                with amp.autocast():
                    try:
                        acc, loss = self.set_forward_loss(x)
                        # Scale loss for accumulation
                        loss = loss / accumulation_steps
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            # Emergency cleanup and retry with CPU offloading
                            torch.cuda.empty_cache()
                            print("Emergency CPU offloading...")
                            # This time we'll process on CPU if needed
                            acc, loss = self.set_forward_loss_cpu_fallback(x)
                            loss = loss / accumulation_steps
                        else:
                            raise e
                
                # Scale and accumulate gradients
                scaler.scale(loss).backward()
                
                # Update weights after accumulation
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    
                avg_loss += loss.item() * accumulation_steps
                avg_acc.append(acc)
                train_pbar.set_description('Epoch {:03d}/{:03d} | Acc {:.6f} | Loss {:.6f}'.format(
                    epoch + 1, num_epoch, np.mean(avg_acc) * 100, avg_loss/float(i+1)))
                train_pbar.update(1)
                
                # Aggressive cleanup every few iterations
                if i % 10 == 0:
                    torch.cuda.empty_cache()
        
        if wandb_flag:
            wandb.log({"Loss": avg_loss/float(i + 1),'Train Acc': np.mean(avg_acc) * 100}, step=epoch + 1)

    def val_loop(self, val_loader, epoch, wandb_flag, record = None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(val_loader)
        with tqdm.tqdm(total=len(val_loader)) as val_pbar:
            for i, (x,_) in enumerate(val_loader):
                if self.change_way:
                    self.n_way  = x.size(0)
                correct_this, count_this = self.correct(x)
                acc_all.append(correct_this / count_this * 100)
                val_pbar.set_description('Validation    | Acc {:.6f}'.format(np.mean(acc_all)))
                val_pbar.update(1)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        if wandb_flag:
            wandb.log({'Val Acc': acc_mean},  step = epoch + 1)
        print('Val Acc = %4.2f%% +- %4.2f%%' %(  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean

    def set_forward_loss_cpu_fallback(self, x):
        """Emergency fallback that offloads computation to CPU when OOM occurs"""
        # Move model to CPU temporarily
        self.cpu()
        torch.cuda.empty_cache()
        
        # Process in smaller chunks on CPU
        if isinstance(x, list):
            x = [item.cpu() for item in x]
        else:
            x = x.cpu()
        
        # Forward pass on CPU
        acc, loss = self.set_forward_loss(x)
        
        # Move model back to GPU
        self.to(device)
        
        return acc, loss