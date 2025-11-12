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

    def parse_feature(self,x,is_feature):
        x    = Variable(x.to(device))
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.k_shot + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.k_shot + self.n_query, *z_all.size()[1:])
            
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
                
                # Add gradient clipping for training stability and better convergence
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                avg_loss += loss.item()
                avg_acc.append(acc)
                train_pbar.set_description('Epoch {:03d}/{:03d} | Acc {:.6f}  | Loss {:.6f}'.format(
                    epoch + 1, num_epoch, np.mean(avg_acc) * 100, avg_loss/float(i+1)))
                train_pbar.update(1)
        if wandb_flag:
            wandb.log({"Loss": avg_loss/float(i + 1),'Train Acc': np.mean(avg_acc) * 100},  step=epoch + 1)

    def val_loop(self, val_loader, epoch, wandb_flag, record = None):
        from sklearn.metrics import f1_score
        
        correct =0
        count = 0
        acc_all = []
        all_preds = []
        all_labels = []
        
        iter_num = len(val_loader)
        with tqdm.tqdm(total=len(val_loader)) as val_pbar:
            for i, (x,_) in enumerate(val_loader):
                if self.change_way:
                    self.n_way  = x.size(0)
                
                # Get predictions for F1 score calculation
                scores = self.set_forward(x)
                pred = scores.data.cpu().numpy().argmax(axis=1)
                y = np.repeat(range(self.n_way), self.n_query)
                
                all_preds.extend(pred.tolist())
                all_labels.extend(y.tolist())
                
                correct_this, count_this = self.correct(x)
                acc_all.append(correct_this / count_this * 100)
                val_pbar.set_description('Validation    | Acc {:.6f}'.format(np.mean(acc_all)))
                val_pbar.update(1)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        
        # Calculate and display per-class F1 scores
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        class_f1 = f1_score(all_labels, all_preds, average=None)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print('Val Acc = %4.2f%% +- %4.2f%%' %(  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        print(f"\n📊 Validation F1 Score Results:")
        print(f"Macro-F1: {macro_f1:.4f}")
        print("\nPer-class F1 scores:")
        for i, f1 in enumerate(class_f1):
            print(f"  Class {i}: {f1:.4f}")
        
        if wandb_flag:
            wandb.log({'Val Acc': acc_mean, 'Val Macro-F1': macro_f1},  step = epoch + 1)

        return acc_mean