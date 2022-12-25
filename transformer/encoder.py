import torch.nn as nn
from . import constant as cf
from .utils import create_source_mask
from .layers import MultiHeadAttention, PositionWiseFeedForward, EncoderEmbeddingLayer, PositionalEncoding


class EncoderLayer(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.Figure 1
    # Single block encoder

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.norm = nn.LayerNorm(cf.d_model, eps=cf.d_model)
        self.attentions = MultiHeadAttention()
        self.ffn = PositionWiseFeedForward()


    def forward(self, x, mask):
        # x = batch_size x seq_len x hidden_dim
        # mask = batch_size x seq_len x seq_len

        norm = self.norm(x)
        attention = self.attentions(norm, norm, norm, mask)[0]
        output = x + attention
        
        norm = self.norm(output)
        ffn = self.ffn(norm)
        output = output + ffn
        # output = batch_size x seq_len x hidden_dim
        return output


class Encoder(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.Figure 1
    # Full encoder block

    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = EncoderEmbeddingLayer()
        self.pos_encoder = PositionalEncoding()
        self.encoders = nn.ModuleList([
            EncoderLayer() for _ in range(cf.n_layer)
        ])
        self.dropout = nn.Dropout(cf.dropout)
        self.norm = nn.LayerNorm(cf.d_model, eps=cf.eps)


    def forward(self, x):
        # x = batch_size x seq_len
        mask = create_source_mask(x)
        # mask = batch_size x source_len x source_len
        x = self.embedding(x) 
        x = self.pos_encoder(x)
        x = self.dropout(x)
        # x = batch_size x seq_len x hidden_dim

        for encoder in self.encoders:
            x = encoder(x, mask)
        # x = batch_size x source_len x hidden_dim
        x = self.norm(x)
        return x
