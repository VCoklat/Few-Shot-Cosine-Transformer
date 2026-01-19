
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate
from methods.optimal_few_shot import OptimizedConv4

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, k_shot, n_query):
        super(ProtoNet, self).__init__(model_func,  n_way, k_shot, n_query)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.k_shot, -1).mean(1) # (N, D)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1) # (N*Q, D)

        dists = euclidean_dist(z_query, z_proto)
        return -dists

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        loss = self.loss_fn(scores, y_query)
        acc = count_acc(scores, y_query)
        return acc, loss

class MatchingNet(MetaTemplate):
    def __init__(self, model_func,  n_way, k_shot, n_query):
        super(MatchingNet, self).__init__(model_func,  n_way, k_shot, n_query)
        self.loss_fn = nn.NLLLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        
        z_support = z_support.contiguous().view(self.n_way * self.k_shot, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Normalize 
        z_support = F.normalize(z_support, dim=1)
        z_query = F.normalize(z_query, dim=1)

        # Cosine similarity
        scores = torch.mm(z_query, z_support.t()) # (Q, S)
        
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        
        # Target for matching net is a bit different if k_shot > 1
        # Propagate labels
        support_y = torch.from_numpy(np.repeat(range(self.n_way), self.k_shot)).cuda()
        
        scores = self.set_forward(x)
        
        # Softmax over support set
        logprobs = F.log_softmax(scores, dim=1)
        
        # Create one-hot for support labels
        y_onehot = torch.zeros(self.n_way * self.k_shot, self.n_way).cuda()
        y_onehot.scatter_(1, support_y.view(-1, 1), 1)

        # Probabilities of classes
        log_p_y = torch.mm(F.softmax(scores, dim=1), y_onehot).log()

        loss = self.loss_fn(log_p_y, y_query)
        acc = count_acc(log_p_y, y_query)
        return acc, loss

class RelationNet(MetaTemplate):
    def __init__(self, model_func,  n_way, k_shot, n_query, hidden_dim=64):
        super(RelationNet, self).__init__(model_func,  n_way, k_shot, n_query)
        
        # Relation module
        # Assuming feature dim is known or passed. If not, we might need to infer.
        # But MetaTemplate doesn't store feature_dim easily unless we run a pass.
        # Hardcoding expectation of a certain size or adding adaptive layer.
        
        # Since we use Conv4 usually, features are [C, H, W] not flattened?
        # RelationNet works on feature maps.
        # parse_feature in MetaTemplate usually flattens if 'flatten=True' in model.
        # We need to enforce flatten=False for RelationNet or handle 1D relations.
        
        # For simplicity/robustness with current codebase which likely flattens:
        # We will implement 1D Relation Module (MLP)
        self.loss_fn = nn.MSELoss() 
        self.hidden_dim = hidden_dim
        
        # We need to know feature dim. 
        # Standard hack: lazy init or just use large enough
        self.relation_module = None 

    def init_relation_module(self, feat_dim):
        self.relation_module = nn.Sequential(
            nn.Linear(feat_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).cuda()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        
        if self.relation_module is None:
             self.init_relation_module(z_support.size(-1))

        # Prototype (SUM or MEAN)
        z_proto = z_support.view(self.n_way, self.k_shot, -1).sum(1) # (N, D)
        z_query = z_query.view(self.n_way * self.n_query, -1) # (Q, D)

        # Relation pairs
        # (Q, N, D*2)
        z_proto_ext = z_proto.unsqueeze(0).repeat(z_query.size(0), 1, 1)
        z_query_ext = z_query.unsqueeze(1).repeat(1, self.n_way, 1)
        
        relation_pairs = torch.cat((z_proto_ext, z_query_ext), 2)
        relation_pairs = relation_pairs.view(-1, z_proto.size(-1)*2)
        
        validity = self.relation_module(relation_pairs)
        validity = validity.view(z_query.size(0), self.n_way)
        
        return validity

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        
        scores = self.set_forward(x)
        
        # MSE Loss with One-Hot
        y_onehot = torch.zeros_like(scores)
        y_onehot.scatter_(1, y_query.view(-1, 1), 1)
        
        loss = self.loss_fn(scores, y_onehot)
        acc = count_acc(scores, y_query)
        return acc, loss

class MetaBaseline(MetaTemplate):
    def __init__(self, model_func,  n_way, k_shot, n_query):
        super(MetaBaseline, self).__init__(model_func,  n_way, k_shot, n_query)
        self.loss_fn = nn.CrossEntropyLoss()
        self.temp = nn.Parameter(torch.tensor(10.0))

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        
        z_proto = z_support.view(self.n_way, self.k_shot, -1).mean(1)
        z_query = z_query.view(self.n_way * self.n_query, -1)
        
        z_proto = F.normalize(z_proto, dim=1)
        z_query = F.normalize(z_query, dim=1)
        
        scores = torch.mm(z_query, z_proto.t()) * self.temp
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)
        acc = count_acc(scores, y_query)
        return acc, loss

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()
