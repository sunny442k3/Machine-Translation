import torch


# batch_size = 32
# max_len = 35
# d_model = 512
# d_ff = 2048
# n_head = 8
# n_layer = 6
# dropout = 0.1
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# warm_steps = 4000
# pad_idx = 0
# eps = 1e-6
# src_vocab_size = 106111
# trg_vocab_size = 22536
# epochs = 20
# clip = 1


batch_size = 128
max_len = 35
d_model = 512
d_ff = 1024
n_head = 8
n_layer = 4
dropout = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warm_steps = 4000
pad_idx = 0
eps = 1e-6
src_vocab_size = 106111
trg_vocab_size = 22536
epochs = 20
clip = 1