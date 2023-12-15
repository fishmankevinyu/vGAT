import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, MultiGraphAttentionLayer
from einops.layers.torch import Rearrange
from einops import repeat

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class GAT_base(nn.Module):
    def __init__(self, n_layers, embed_dim, nhid, nclass, dropout, alpha, nheads, concat):
        """Dense version of GAT."""
        super(GAT_base, self).__init__()
        image_size = 224
        patch_size = 28
        channel_size = 3

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width
        # - start with a patch_embedding layer to divide image into patches and
        #   convert each patch into vector embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim * channel_size),
            nn.Linear(patch_dim * channel_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.dropout = dropout
        self.multiAttentions = [MultiGraphAttentionLayer(embed_dim, nhid, nheads, 
                                                    dropout=dropout, 
                                                    alpha=alpha, 
                                                    concat=concat)]
        for _ in range(n_layers - 1):
            if concat:
                self.multiAttentions.append(MultiGraphAttentionLayer(nheads * nhid, nhid, nheads, 
                                                    dropout=dropout, 
                                                    alpha=alpha, 
                                                    concat=concat))
            else:
                self.multiAttentions.append(MultiGraphAttentionLayer(nhid, nhid, nheads, 
                                                    dropout=dropout, 
                                                    alpha=alpha, 
                                                    concat=concat))

        for i, attention_layer in enumerate(self.multiAttentions):
            self.add_module('multi-attention_{}'.format(i), attention_layer)
        #need to aggregate all nodes info
        if concat:
            self.out_att = MultiGraphAttentionLayer(nhid * nheads, nhid, 1, dropout=dropout, alpha=alpha, concat=False)
        else:
            self.out_att = MultiGraphAttentionLayer(nhid, nhid, 1, dropout=dropout, alpha=alpha, concat=False)
        self.linear = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        x = self.to_patch_embedding(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for attentions_layer in self.multiAttentions:
            x = attentions_layer(x, adj)
        x = self.out_att(x, adj)
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
class GAT_cls(nn.Module):
    def __init__(self, n_layers, embed_dim, nhid, nclass, dropout, alpha, nheads, concat):
        """Dense version of GAT."""
        super(GAT_cls, self).__init__()
        image_size = 224
        patch_size = 28
        channel_size = 3
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # - start with a patch_embedding layer to divide image into patches and
        #   convert each patch into vector embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim * channel_size),
            nn.Linear(patch_dim * channel_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.dropout = dropout
        self.multiAttentions = [MultiGraphAttentionLayer(embed_dim, nhid, nheads, 
                                                    dropout=dropout, 
                                                    alpha=alpha, 
                                                    concat=concat)]
        for _ in range(n_layers - 1):
            if concat:
                self.multiAttentions.append(MultiGraphAttentionLayer(nheads * nhid, nhid, nheads, 
                                                    dropout=dropout, 
                                                    alpha=alpha, 
                                                    concat=concat))
            else:
                self.multiAttentions.append(MultiGraphAttentionLayer(nhid, nhid, nheads, 
                                                    dropout=dropout, 
                                                    alpha=alpha, 
                                                    concat=concat))

        for i, attention_layer in enumerate(self.multiAttentions):
            self.add_module('multi-attention_{}'.format(i), attention_layer)
        #need to aggregate all nodes info
        if concat:
            self.out_att = MultiGraphAttentionLayer(nhid * nheads, nhid, 1, dropout=dropout, alpha=alpha, concat=False)
        else:
            self.out_att = MultiGraphAttentionLayer(nhid, nhid, 1, dropout=dropout, alpha=alpha, concat=False)
        self.linear = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        for attentions_layer in self.multiAttentions:
            x = attentions_layer(x, adj)
        x = self.out_att(x, adj)
        x = x[:, 0]
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
class GAT_pos(nn.Module):
    def __init__(self, n_layers, embed_dim, nhid, nclass, dropout, alpha, nheads, concat, cls=False):
        """Dense version of GAT."""
        super(GAT_pos, self).__init__()
        image_size = 224
        patch_size = 28
        channel_size = 3

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width

        self.cls = cls
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        # - start with a patch_embedding layer to divide image into patches and
        #   convert each patch into vector embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim * channel_size),
            nn.Linear(patch_dim * channel_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.dropout = dropout
        self.multiAttentions = [MultiGraphAttentionLayer(embed_dim, nhid, nheads, 
                                                    dropout=dropout, 
                                                    alpha=alpha, 
                                                    concat=concat)]
        for _ in range(n_layers - 1):
            if concat:
                self.multiAttentions.append(MultiGraphAttentionLayer(nheads * nhid, nhid, nheads, 
                                                    dropout=dropout, 
                                                    alpha=alpha, 
                                                    concat=concat))
            else:
                self.multiAttentions.append(MultiGraphAttentionLayer(nhid, nhid, nheads, 
                                                    dropout=dropout, 
                                                    alpha=alpha, 
                                                    concat=concat))

        for i, attention_layer in enumerate(self.multiAttentions):
            self.add_module('multi-attention_{}'.format(i), attention_layer)
        #need to aggregate all nodes info
        if concat:
            self.out_att = MultiGraphAttentionLayer(nhid * nheads, nhid, 1, dropout=dropout, alpha=alpha, concat=False)
        else:
            self.out_att = MultiGraphAttentionLayer(nhid, nhid, 1, dropout=dropout, alpha=alpha, concat=False)
        self.linear = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = F.dropout(x, self.dropout, training=self.training)
        for attentions_layer in self.multiAttentions:
            x = attentions_layer(x, adj)
        x = self.out_att(x, adj)

        if self.cls:
            x = x[:, 0]
        else:
            x = torch.mean(x, dim=1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
   

