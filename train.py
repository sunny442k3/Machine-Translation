import pickle
import torch
from transformer import constant as cf
from transformer.model import Transformer

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)
    model = Transformer().to(cf.device)
    train_loader = open("./checkpoint/train_loader.pickle", "rb")
    train_loader = pickle.load(train_loader)
    print(len(train_loader))
    print(cf.device)
    model.eval()
    for idx, (source, target) in enumerate(train_loader):
        source = source.long().to(cf.device)
        target = target.long().to(cf.device)

        output = model(source, target[:, :-1])[0]
        print(output.size())

        break