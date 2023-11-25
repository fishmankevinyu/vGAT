import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    #the original input is 2d matrix, where each data is one node
    #the new input is 3d, where each data is one graph
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        #new class variable
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        #extend this to multiple Ws
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        #extend this to multiple as
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.mm(h, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vecs = -9e15*torch.ones_like(e)

        attentions = torch.where(adj > 0, e, zero_vecs)
        attentions = F.softmax(attentions, dim=1)
        attentions = F.dropout(attentions, self.dropout, training=self.training)

        h_prime = torch.matmul(attentions, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
                # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class MultiGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, nheads, dropout, alpha, concat=True):
        super(MultiGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        #new class variable
        self.in_features = in_features
        self.alpha = alpha
        self.concat = concat
        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=concat) 
                            for _ in range(nheads)]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, h, adj):
        B,H,D = h.shape
        x = h.view(B * H, D)
        if self.concat:
            x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        else:
            x = torch.mean(torch.stack([F.elu(att(x, adj)) for att in self.attentions]), dim=0)
        x = x.view(B, H, -1)

        return x