import torch
from torch import nn
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import math
from torch.autograd import Variable

# # initialize parameters (important)
# for p in self.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)


class TransEncoder(nn.Module):
    '''
    embed layer, transformer stack
    feat_input_dim: input feature dim
    m_output_dim: memory output dim
    '''

    def __init__(self, feat_input_dim, mem_output_dim, n=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.1):
        super(TransEncoder, self).__init__()
        attn = MultiHeadAttention(n_heads, d_model)
        ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.embed_layer = nn.Linear(feat_input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.encoder = TransStack(TransformerBlock(d_model, deepcopy(attn), None, deepcopy(ffn), dropout), n)
        self.output_layer = nn.Linear(d_model, mem_output_dim)

    def forward(self, feat_inp, feat_mask, with_pe):
        # feat_inp=[bs,m,dim]
        mem_ebedding = self.embed_layer(feat_inp)  # embedding (lookup table)
        if with_pe:  # positional encoding
            mem_ebedding = self.pos_enc(mem_ebedding)
        mem_output = self.encoder(mem_ebedding, feat_mask, None, None)
        memory = self.output_layer(mem_output)
        return memory  # [bs,m,dim]


class TransDecoder(nn.Module):
    '''
    embed layer, transformer stack, output layer
    tgt_input_dim: target query input dim
    tgt_output_dim: target query output dim
    '''

    def __init__(self, tgt_input_dim, tgt_output_dim, n=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.1):
        super(TransDecoder, self).__init__()
        attn = MultiHeadAttention(n_heads, d_model)
        ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.embed_layer = nn.Linear(tgt_input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.decoder = TransStack(TransformerBlock(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ffn), dropout), n)
        self.output_layer = nn.Linear(d_model, tgt_output_dim)

    def forward(self, memory, m_mask, tgt, tgt_mask, with_pe):
        # memory=[bs,m,dim], tgt=[bs,n,dim]
        tgt_embedding = self.embed_layer(tgt)  # embedding (lookup table)
        if with_pe:  # positional encoding  todo: memory是不是也要加pe
            tgt_embedding = self.pos_enc(tgt_embedding)
        tgt_output = self.decoder(tgt_embedding, tgt_mask, memory, m_mask)
        final_output = self.output_layer(tgt_output)
        return final_output  # [bs,n,dim]


class TransStack(nn.Module):
    '''
    stack N transformer blocks with cross_attn
    t_block: nn.Module
    '''
    def __init__(self, t_block, n):
        super(TransStack, self).__init__()
        self.layers = clones(t_block, n)
        self.norm = LayerNorm(t_block.d_model)

    def forward(self, x, x_mask, m, m_mask):
        for layer in self.layers:
            x = layer(x, x_mask, m, m_mask)
        return self.norm(x)



class TransformerBlock(nn.Module):
    """
    self_attn, cross_attn (optimal), ffn
    """

    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.norm_layers = clones(LayerNorm(d_model), 3)
        self.droput = nn.Dropout(dropout)

    def forward(self, x, x_mask, m, m_mask):
        # self_attn
        if self.self_attn is not None:
            x = self.norm_layers[0](x + self.droput(self.self_attn(x, x, x, x_mask)))

        # cross_attn
        if self.cross_attn is not None:
            x = self.norm_layers[1](x + self.droput(self.cross_attn(x, m, m, m_mask)))

        # ffn
        x = self.norm_layers[2](x + self.droput(self.feed_forward(x)))
        return x


class MultiHeadAttention(nn.Module):
    '''
    Multi-Heads Attention
    '''

    def __init__(self, n_heads, d_model, dropout=0.1):
        """
        n_heads: number of heads
        d_model: model dim
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, value=-1e9)  # mask the False elements
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    """
    a residual connection block with a layer norm and a dropout
    """

    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # sublayer: nn.Module
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """
    the PE function, it can be added to both the encoder and the decoder before input
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings in log space.
        pe = torch.zeros(max_len, d_model)  # [len,dim]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1,len,dim] where len indicate position
        self.register_buffer('pe', pe)  # parameters not updated

    def forward(self, x):
        # x = [bs,n,dim] add the positional encoding to the original inputs
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  # pe -> [bs,n,dim]
        return self.dropout(x)


class LayerNorm(nn.Module):
    """
    a LayerNorm layer
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def clones(module, n):
    """
    stack N identical layers using nn.ModuleList
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])