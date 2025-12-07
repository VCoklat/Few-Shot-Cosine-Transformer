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
import inspect
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore")


def call_model_func(model_func, dataset='miniImagenet', feti=0, flatten=True):
    """
    Helper function to call model_func with proper arguments.
    
    Handles multiple patterns:
    1. Closure pattern (old code): model_func is a lambda/closure with no params
    2. Conv4/Conv6 pattern: model_func(dataset, flatten=True)
    3. ResNet pattern: model_func(FETI, dataset, flatten=True)
    
    Args:
        model_func: Function or closure that returns a model
        dataset: Dataset name to pass to model_func if needed
        feti: FETI parameter for ResNet models
        flatten: Whether to flatten output (method-specific: True for transformers, False for CTX)
        
    Returns:
        Instantiated model
        
    Raises:
        TypeError: If model_func cannot be called with any known signature
    """
    try:
        sig = inspect.signature(model_func)
        params = list(sig.parameters.keys())
        
        # Check parameter count and names to determine calling pattern
        if len(params) == 0:
            # Closure with no parameters
            return model_func()
        elif len(params) >= 2:
            # Check if first parameter is 'FETI' (ResNet pattern)
            if params[0].upper() == 'FETI':
                # ResNet pattern: (FETI, dataset, flatten=True)
                if 'flatten' in params:
                    return model_func(feti, dataset, flatten=flatten)
                else:
                    return model_func(feti, dataset)
            elif 'dataset' in params:
                # Conv4/Conv6 pattern: (dataset, flatten=True)
                if 'flatten' in params:
                    return model_func(dataset=dataset, flatten=flatten)
                else:
                    return model_func(dataset=dataset)
        elif len(params) == 1:
            # Single parameter, likely 'dataset'
            if 'dataset' in params:
                return model_func(dataset=dataset)
        
        # If we couldn't match any pattern above, raise informative error
        raise TypeError(f"Cannot determine how to call model_func with parameters: {params}")
        
    except (ValueError, TypeError) as e:
        # Fallback: try different calling patterns
        try:
            # Try closure pattern first (most common in old code)
            return model_func()
        except TypeError:
            try:
                # Try Conv4 pattern with keyword
                return model_func(dataset=dataset)
            except TypeError:
                try:
                    # Try ResNet pattern
                    return model_func(feti, dataset)
                except TypeError:
                    # Re-raise original error with context
                    raise TypeError(f"Cannot call model_func with any known signature. "
                                  f"Parameters detected: {sig.parameters if 'sig' in locals() else 'unknown'}. "
                                  f"Original error: {e}") from e


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, k_shot, n_query, change_way = True, dataset='miniImagenet', feti=0, flatten=True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.k_shot     = k_shot
        self.n_query    = n_query
        
        # Handle both closure (no args) and function (requires dataset) patterns
        self.feature = call_model_func(model_func, dataset=dataset, feti=feti, flatten=flatten)
        
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

    def parse_feature(self,x,is_feature):
        x    = Variable(x.to(device))
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.k_shot + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            z_all       = z_all.reshape( self.n_way, self.k_shot + self.n_query, *z_all.size()[1:])
            
        z_support   = z_all[:, :self.k_shot]
        z_query     = z_all[:, self.k_shot:]

        return z_support, z_query

    def correct(self, x):       
        scores = self.set_forward(x)
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.to(device))

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels
        top1_correct = (topk_ind[:,0] == y_query).sum().item()
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, num_epoch, train_loader, wandb_flag, optimizer):
        avg_loss = 0
        avg_acc = []
        with tqdm.tqdm(total = len(train_loader)) as train_pbar:
            for i, (x, _) in enumerate(train_loader):        
                if self.change_way:
                    self.n_way  = x.size(0)
                
                optimizer.zero_grad()
                acc, loss = self.set_forward_loss(x = x.to(device))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                avg_acc.append(acc)
                train_pbar.set_description('Epoch {:03d}/{:03d} | Acc {:.6f}  | Loss {:.6f}'.format(
                    epoch + 1, num_epoch, np.mean(avg_acc) * 100, avg_loss/float(i+1)))
                train_pbar.update(1)
        if wandb_flag:
            wandb.log({"Loss": avg_loss/float(i + 1),'Train Acc': np.mean(avg_acc) * 100},  step=epoch + 1)

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