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


def get_loader(path, batch_size=128):
    df = read_csv("./dataset/corpus.csv")
    kor_word = df["korean"].values 
    eng_word = df["english"].values 

    kor_tokenizer = Tokenizer(kor_word, 150000)
    eng_tokenizer = Tokenizer(eng_word, 50000)

    df = read_csv(path)
    kor_word = df["korean"].values 
    eng_word = df["english"].values 
    kor_data = make_dataset(kor_word, kor_tokenizer)
    eng_data = make_dataset(eng_word, eng_tokenizer)
    kor_data = torch.tensor(kor_data).float()
    eng_data = torch.tensor(eng_data).float()
    dataset = CustomDataset(kor_data, eng_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    return dataloader


if __name__ == "__main__":
    en_train = None 
    vi_train = None
    en_train = open("./dataset/train-en-vi/train.en", "rb").readlines()
    en_train = [i.decode("utf8") for i in en_train]
    vi_train = open("./dataset/train-en-vi/train.vi", "r", encoding="utf8").readlines()
    print(en_train[:10])
    print(vi_train[:10])