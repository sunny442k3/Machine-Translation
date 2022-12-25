import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, source, target):
        # source = batch_size x source_len
        # target = batch_size x target_len

        encoder_output = self.encoder(source)
        # encoder_output = batch_size x source_len x hidden_dim
        output, attention_map = self.decoder(target, source, encoder_output)
        # output = batch_size x target_len x output_dim
        return output, attention_map