import torch
import pickle
import pandas as pd 
import transformer.constant as cf
from transformer.tokenizer import Tokenizer
from torch.utils.data import DataLoader





def read_csv(path):
    df = pd.read_csv(path, encoding='utf-8')
    return df 


def make_dataset(sequences, tokenizer):
    tokens = []
    for seq in sequences:
        token = tokenizer.encode(seq, cf.max_len)
        tokens.append(token)
    return tokens


class CustomDataset:

    def __init__(self, source, target):
        self.source = source 
        self.target = target
    

    def __len__(self):
        return len(self.source)

    
    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


def get_loader(src, trg, src_tokenizer, trg_tokenizer, batch_size=128, test_size=0.3):
    src_data = make_dataset(src, src_tokenizer)
    trg_data = make_dataset(trg, trg_tokenizer)
    src_data = torch.tensor(src_data).float()
    trg_data = torch.tensor(trg_data).float()
    lim = int(test_size*len(src))
    valid_dataset = CustomDataset(src_data[:lim], trg_data[:lim])
    train_dataset = CustomDataset(src_data[lim:], trg_data[lim:])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    return train_loader, valid_loader
