import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from vgae.models import VGAE
import numpy as np
from torch_geometric.nn import GCNConv
class Linear(nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return x

class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        out = self.Linear1(x)
        return torch.log_softmax(out, dim=1), out


class VGAE(nn.Module):
    def __init__(self, dim_in, dim_h, dim_z, gae):
        super(VGAE,self).__init__()
        self.dim_z = dim_z
        self.gae = gae
        self.base_gcn = GraphConvSparse(dim_in, dim_h)
        self.gcn_mean = GraphConvSparse(dim_h, dim_z, activation=False)
        self.gcn_logstd = GraphConvSparse(dim_h, dim_z, activation=False)

    def encode(self, X,A):
        hidden = self.base_gcn(X,A)
        self.mean = self.gcn_mean(hidden,A)
        if self.gae:
            # graph auto-encoder
            return self.mean
        else:
            # variational graph auto-encoder
            self.logstd = self.gcn_logstd(hidden,A)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
            return sampled_z

    def decode(self, Z):
        A_pred = Z @ Z.T
        return A_pred

    def forward(self, X,A):
        Z = self.encode(X,A)
        A_pred = self.decode(Z)
        return A_pred

class DDPT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, use_bn = False):
        super(DDPT, self).__init__()

        self.encoder = MLP_encoder(nfeat=nfeat,
                                 nhid=nhid,
                                 dropout=dropout)
        

        # self.edgepredicter = VGAE(adj=adj,dim_in=nfeat,dim_h=128,dim_z=256,gae=False)


        self.classifier = MLP_classifier(nfeat=nhid,
                                         nclass=nclass,
                                         dropout=dropout)

        

        self.proj_head1 = Linear(nhid, nhid, dropout, bias=True)

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid)

    def forward(self, features, eval = False):
        if self.use_bn:
            features = self.bn1(features)
        query_features = self.encoder(features)
        if self.use_bn:
            query_features = self.bn2(query_features)

        output, emb = self.classifier(query_features)
        if not eval:
            emb = self.proj_head1(query_features)
        return emb, output
    

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=True):
        super(GraphConvSparse, self).__init__()
        self.weight = self.glorot_init(input_dim, output_dim)

        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
        return nn.Parameter(initial)

    def forward(self, inputs,adj):
        x = inputs @ self.weight
        x = adj @ x
        if self.activation:
            return F.elu(x)
        else:
            return x


import torch 
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def  __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):    #x,adj
        x = F.relu(self.gc1(x, adj))
        h = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return h,F.log_softmax(x, dim=1)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input@self.weight
        # print(support)
        # print(adj)
        output = adj@support
        if self.bias is not None:
            return output + self.bias
        else:
            return output