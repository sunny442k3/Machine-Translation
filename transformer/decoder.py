import torch
import torch.nn as nn
from . import constant as cf
from .utils import create_target_mask
from .layers import MultiHeadAttention, PositionWiseFeedForward, DecoderEmbeddingLayer, PositionalEncoding


class DecoderLayer(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.Figure 1
    # Single block decoder

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.norm = nn.LayerNorm(cf.d_model, eps=1e-6)
        self.attentions = MultiHeadAttention()
        self.encoder_attentions = MultiHeadAttention()
        self.ffn = PositionWiseFeedForward()


    def forward(self, target, encoder_output, target_mask, decoder_encoder_mask):
        # target = batch_size x target_len x hidden_dim
        # encoder_output = batch_size x source_len x hidden_dim
        # target_mask = batch_size x target_len x target_len
        # decoder_encoder_mask = batch_size x target_len x source_len

        norm_target = self.norm(target)
        attention_target = self.attentions(norm_target, norm_target, norm_target, target_mask)[0]
        output = target + attention_target

        norm_output = self.norm(output)
        attention_output, attention_map = self.attentions(norm_output, encoder_output, encoder_output, decoder_encoder_mask)
        output = output + attention_output

        norm_output = self.norm(output)
        ffn = self.ffn(norm_output)
        output = output + ffn 
        # output = batch_size x target_len x hidden_dim
        return output, attention_map


class Decoder(nn.Module):

    # https://arxiv.org/pdf/1706.03762.pdf - 3.Figure 1
    # Full decoder block

    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = DecoderEmbeddingLayer()
        self.pos_encoder = PositionalEncoding()
        self.decoders = nn.ModuleList([
            DecoderLayer() for _ in range(cf.n_layer)
        ])
        self.dropout = nn.Dropout(cf.dropout)
        self.norm = nn.LayerNorm(cf.d_model, eps=cf.eps)


    def forward(self, target, source, encoder_output):
        # target = batch_size x target_len
        # source = batch_size x source_len
        # encoder_output = batch_size x source_len x hidden_dim
        target_mask, decoder_encoder_mask = create_target_mask(source, target)
        
        target = self.embedding(target)
        target = self.pos_encoder(target)
        target = self.dropout(target)
        # target = batch_size x target_len x hidden_dim

        for decoder in self.decoders:
            target, attention_map = decoder(target, encoder_output, target_mask, decoder_encoder_mask)
        # target = batch_size x target_len x hidden_dim
        
        target = self.norm(target)
        output = torch.matmul(target, self.embedding.emb_layer.weight.transpose(0, 1))
        # output = batch_size x target_len x output_dim
        return output, attention_map