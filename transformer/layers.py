import math
import torch 
import numpy as np
import torch.nn as nn
from . import constant as cf
import torch.nn.functional as F


class SelfAttention(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.2.1

    def __init__(self):
        super(SelfAttention, self).__init__()
        self.hidden_dim = cf.d_model
        self.att_dim = cf.d_model // cf.n_head

        self.wq = nn.Linear(self.hidden_dim, self.att_dim, bias=False)
        self.wk = nn.Linear(self.hidden_dim, self.att_dim, bias=False)
        self.wv = nn.Linear(self.hidden_dim, self.att_dim, bias=False)

        self.dropout = nn.Dropout(cf.dropout)
        self.init_weight()

    
    def init_weight(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        if self.wq.bias is not None:
            nn.init.constant_(self.wq.bias, 0)
        if self.wk.bias is not None:
            nn.init.constant_(self.wk.bias, 0)
        if self.wv.bias is not None:
            nn.init.constant_(self.wv.bias, 0)


    def forward(self, q, k, v, mask=None):
        # q, k, v = batch_size x seq_len x hidden_dim
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # q, k, v = batch_size x seq_len x attention_dim

        self_att = q @ k.transpose(1,2)
        self_att /= self.att_dim
        # self att = batch_size x seq_len x seq_len

        if mask is not None:
            self_att = self_att.masked_fill(mask, -np.inf)
        
        self_att = F.softmax(self_att, dim=-1)
        self_att = self.dropout(self_att)

        att = self_att @ v 
        # att = batch_size x seq_len x attention_dim
        return self.dropout(att), self_att


class MultiHeadAttention(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.2.2

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        assert cf.d_model % cf.n_head == 0, "Hidden dim should be devide by n_head"
        self.hidden_dim = cf.d_model
        self.n_heads = cf.n_head
        self.attentions = nn.ModuleList([
            SelfAttention() for _ in range(self.n_heads)
        ])
        self.wo = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(cf.dropout)
        # self.init_weight()

    
    def init_weight(self):
        nn.init.xavier_uniform_(self.wo)
        if self.wo.bias is not None:
            nn.init.constant_(self.wo.bias, 0)
        

    def forward(self, q, k, v, mask=None):
        # q, k, v = batch_size x seq_len x hidden_dim
        heads = [
            attension(q, k, v, mask) for attension in self.attentions
        ]
        weights = [att[0] for att in heads]
        attentions = [att[1] for att in heads]

        weights = torch.cat(weights, dim=-1)
        output = self.wo(weights)
        output = self.dropout(output)
        # output = batch_size x seq_len x hidden_dim
        return output, attentions
        

class PositionWiseFeedForward(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.3

    def __init__(self):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(cf.d_model, cf.d_ff, bias=True)
        self.w2 = nn.Linear(cf.d_ff, cf.d_model, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(cf.dropout)
        # self.init_weight()


    def init_weight(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        if self.w1.bias is not None:
            nn.init.constant_(self.w1.bias, 0)
        if self.w2.bias is not None:
            nn.init.constant_(self.w2.bias, 0)


    def forward(self, x):
        output = self.w1(x)
        output = self.activation(output)
        output = self.w2(output)
        output = self.dropout(output)
        return output


class EncoderEmbeddingLayer(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.4

    def __init__(self):
        super(EncoderEmbeddingLayer, self).__init__()
        self.emb_layer = nn.Embedding(cf.src_vocab_size, cf.d_model, padding_idx=cf.pad_idx)
        nn.init.normal_(self.emb_layer.weight, mean=0, std= 1 / (cf.d_model**(0.5)))
        self.scale = cf.d_model ** (0.5)


    def forward(self, x):
        output = self.emb_layer(x) + self.scale
        return output


class DecoderEmbeddingLayer(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.4

    def __init__(self):
        super(DecoderEmbeddingLayer, self).__init__()
        self.emb_layer = nn.Embedding(cf.trg_vocab_size, cf.d_model, padding_idx=cf.pad_idx)
        nn.init.normal_(self.emb_layer.weight, mean=0, std= 1 / (cf.d_model**(0.5)))
        self.scale = cf.d_model ** (0.5)


    def forward(self, x):
        output = self.emb_layer(x) + self.scale
        return output


class PositionalEncoding(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.5

    def __init__(self):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(cf.max_len, cf.d_model)
        position = torch.arange(0, cf.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, cf.d_model, 2).float() * (-math.log(10000.0) / cf.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)


    def forward(self, x):
        return x + self.pe
