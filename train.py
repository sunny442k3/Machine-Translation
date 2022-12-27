import time
import torch
from transformer import constant as cf
from transformer.model import Transformer
from transformer.scheduler import SchedulerAdam


from transformer.tokenizer import Tokenizer
from preprocessing import get_loader, read_csv, process_dataset, make_dataset


def print_progress(index, total, fi="", last=""):
    percent = ("{0:.1f}").format(100 * ((index) / total))
    fill = int(30 * (index / total))
    spec_char = ["\x1b[1;36;40m━\x1b[0m", "\x1b[1;37;40m━\x1b[0m"]
    bar = spec_char[0]*fill + spec_char[1]*(30-fill)

    percent = " "*(5-len(str(percent))) + str(percent)

    if index == total:
        print("", end="\r")
        print(fi + " " + bar + " " + percent + "% " + last)
    else:
        print("", end="\r")
        print(fi + " " + bar + " " + percent + "% " + last, end="")


class Trainer:

    def __init__(self):
        self.device = cf.device
        self.epochs = cf.epochs
        self.acc = {
            "train": [],
            "valid": []
        }
        self.loss = {
            "train": [],
            "valid": []
        }
        self.init_model()


    def init_model(self):
        self.model = Transformer().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = SchedulerAdam(optimizer)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=cf.pad_idx).to(self.device)


    def load_model(self, path):
        params = torch.load(path)
        self.init_model()
        self.model.load_state_dict(params["model"])
        self.optimizer.load_state_dict(params["optimizer"])
        self.acc, self.loss = params["acc"], params["loss"]


    def save_model(self, path):
        params = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "acc": self.acc,
            "loss": self.loss
        }
        torch.save(params, path)


    def forward_step(self, data_loader, mode="test"):
        if mode == "test":
            self.model.eval()
        else:
            self.model.train()

        loss_his = []
        for idx, (source, target) in enumerate(data_loader):
            if mode == "train":
                self.optimizer.zero_grad()
            source = source.long().to(self.device)
            target = target.long().to(self.device)

            output = self.model(source, target[:, :-1])[0]
            output = output.contiguous().view(-1, output.shape[-1])
            target = target[:, 1:].contiguous().view(-1)
            # output = batch_size * (target_len -1) x output_size
            # target = batch_size * (target_len -1)

            loss = self.criterion(output, target)

            if mode == "train":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cf.clip)
                loss.backward()
                self.optimizer.step()

            loss_his.append(loss.item())

            print_progress(idx+1, len(data_loader), fi=["\t[Train]" if mode!="test" else "\t[Valid]"][0] + " {:3d}/{:3d} batches".format(idx+1, len(data_loader)), last="Loss: {:.5f} lr: {:.7f}".format(loss.item(), self.optimizer.get_lr()))
        print()
        loss = sum(loss_his) / len(loss_his)
        if mode == "test":
            self.loss["valid"].append(loss)
        else:
            self.loss["train"].append(loss)
        return loss


    def fit(self, train_loader, valid_loader, checkpoint):
        print("RUNNING ON:", torch.cuda.get_device_name(0))
        epoch_start_time = time.time()
        for epoch in range(self.epochs):
            print(" 🦄 Start of epoch", epoch+1, "/",self.epochs)
            try:
                train_loss = self.forward_step(train_loader, mode="train")
                valid_loss = self.forward_step(valid_loader, mode="test")
                self.save_model(checkpoint)
                print("\tTrain loss: {:5.5f}".format(train_loss))
                print("\tValid loss: {:5.5f}".format(valid_loss))
                print("\t⏰ Time: {:5.6f}s".format(time.time() - epoch_start_time))
                print()
                epoch_start_time = time.time()
            except KeyboardInterrupt:
                self.save_model(checkpoint)
                return 


if __name__ == "__main__":
    corpus_df = read_csv("./dataset/corpus.csv")
    src_tokenizer = Tokenizer(corpus_df["korean"].values)
    trg_tokenizer = Tokenizer(corpus_df["english"].values)

    df = read_csv("./dataset/train.csv")
    src, trg, missing = process_dataset(df.values)
    print(f"src size: {len(src)} - trg size: {len(trg)} - miss size: {len(missing)}")
    print(f"max length src sequence: {max([len(seq) for seq in src])}")
    print(f"max length trg sequence: {max([len(seq) for seq in trg])}")

    train_loader, valid_loader = get_loader(
        src,
        trg,
        src_tokenizer,
        trg_tokenizer
    )

    print(f"size train loader: {len(train_loader)} - size valid loader: {len(valid_loader)}")

    model = Trainer()
    # model.load_model("./checkpoint/model.pth")
    model.epochs = 5
    model.fit(train_loader, valid_loader, "./checkpoint/model.pth")