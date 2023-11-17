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
    def __init__(self, bSize, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        #new class variable
        self.bSize = bSize
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        #extend this to multiple Ws
        self.Ws = nn.Parameter(torch.empty(size=(bSize, in_features, out_features)))
        nn.init.xavier_uniform_(self.Ws.data, gain=1.414)

        #extend this to multiple as
        self.a = nn.Parameter(torch.empty(size=(bSize, 2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        #extend to Whs with n * W
        #Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        print(np.shape(h))
        print(np.shape(self.Ws[0]))
        Whs = [torch.mm(h[i, :, :], self.Ws[i]) for i in range(self.bSize)]
        print("adj")
        print(adj.shape)
        #series of e's
        es = self._prepare_attentional_mechanism_input(Whs)
        print("es shape: ") 
        print(es.shape)
        zero_vecs = torch.stack([-9e15*torch.ones_like(es[i]) for i in range(self.bSize)])

        attentions = torch.stack([torch.where(adj > 0, es[i], zero_vecs[i]) for i in range(self.bSize)])
        print("attentions shape: ")
        print(attentions.shape)
        attentions = torch.stack([F.softmax(attentions[i], dim=1) for i in range(self.bSize)])
        attentions = torch.stack([F.dropout(attentions[i], self.dropout, training=self.training) for i in range(self.bSize)])
        hs_prime = torch.stack([torch.matmul(attentions[i], Whs[i]) for i in range(self.bSize)])
        print("hs_prime")
        print(hs_prime.shape)

        if self.concat:
            return torch.stack([F.elu(hs_prime[i]) for i in range(self.bSize)])
        else:
            return hs_prime

    def _prepare_attentional_mechanism_input(self, Whs):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Whs1 = torch.stack([torch.matmul(Whs[i], self.a[i, :self.out_features, :]) for i in range(self.bSize)])
        Whs2 = torch.stack([torch.matmul(Whs[i], self.a[i, self.out_features:, :]) for i in range(self.bSize)])        # broadcast add
        es = torch.stack([Whs1[i] + Whs2[i].T for i in range(self.bSize)])
        return self.leakyrelu(es)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
