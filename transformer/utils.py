import torch
from . import constant as cf


def create_subseq_mask(x):
    batch_size, target_len = x.size()
    subseq_mask = torch.triu(torch.ones(target_len, target_len), diagonal=1).bool().to(cf.device)
    # subseq_mask = target_len x target_len 
    mask = subseq_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return mask 


def create_source_mask(x):
    source_len = x.size(1)
    mask = (x == cf.pad_idx)
    mask = mask.unsqueeze(1).repeat(1, source_len, 1)
    # mask = batch_size x source_len x source_len
    return mask


def create_target_mask(x, y):
    target_len = y.size(1)
    source_mask = (x == cf.pad_idx)
    target_mask = (y == cf.pad_idx)
    # target_mask = batch_size x target_len
    # source_mask = batch_size x source_len

    subseq_mask = create_subseq_mask(y)
    # subseq_mask = batch_size x target_len x target_len

    decoder_encoder_mask = source_mask.unsqueeze(1).repeat(1, target_len, 1)
    target_mask = target_mask.unsqueeze(1).repeat(1, target_len, 1)
    target_mask |= subseq_mask
    # target_mask = batch_size x target_len x target_len
    return target_mask, decoder_encoder_mask