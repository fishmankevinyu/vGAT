import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class GAT(nn.Module):
    def __init__(self, bSize, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        image_size = 224
        patch_size = 14
        channel_size = 3
        dim = 128

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width
        # - start with a patch_embedding layer to divide image into patches and
        #   convert each patch into vector embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim * channel_size),
            nn.Linear(patch_dim * channel_size, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(bSize, dim, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        #need to aggregate all nodes info
        self.out_att = GraphAttentionLayer(bSize, nhid, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.linear = nn.Linear(num_patches * nclass, nclass)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.to_patch_embedding(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mean(torch.stack([att(x, adj) for att in self.attentions]), dim=1)
        print("average pooling")
        print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        print("output")
        print(x.shape)
        x = self.leakyrelu(self.linear(x.flatten(start_dim=1)))
        x = F.softmax(x, dim=1)
        print("pred")
        print(x)
        return x

