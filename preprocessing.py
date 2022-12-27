import re
import torch
import pickle
import pandas as pd 
import transformer.constant as cf
from transformer.tokenizer import Tokenizer
from torch.utils.data import DataLoader


def _clean_text(text):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', text)
    return text


def read_csv(path):
    df = pd.read_csv(path, encoding='utf-8')
    return df 


def process_dataset(dataset):
    src_seqs = []
    trg_seqs = []
    missing = []
    for row in dataset:
        src, trg = row
        src = src.split(".")
        trg = trg.split(".")
        if len(src) != len(trg):
            missing.append(row)
            continue 
        src_seqs += src 
        trg_seqs += trg 
    
    return src_seqs, trg_seqs, missing



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


def get_loader(src, trg, src_tokenizer, trg_tokenizer, batch_size=cf.batch_size, test_size=0.3):
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
    

# if __name__ == "__main__":
#     df = read_csv("./dataset/train.csv")
#     src, trg, missing = process_dataset(df.values)
#     print(len(src), len(trg), len(missing))
#     print(max([len(i) for i in src]), max([len(i) for i in trg]))

#     corpus_df = read_csv("./dataset/corpus.csv")
#     src_tokenizer = Tokenizer(corpus_df["korean"].values)
#     trg_tokenizer = Tokenizer(corpus_df["english"].values)
#     print(len(src_tokenizer.token), len(trg_tokenizer.token))