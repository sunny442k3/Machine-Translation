import torch


# HIDDEN_DIM = 128
# N_HEADS = 4
# DROPOUT = 0.1
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ENCODE_LAYER = 6
# DECODE_LAYER = 6
# DFF = 1024
# PADDING_IDX = 0
# MAX_LEN = 500
# INPUT_DIM = 300
# OUTPUT_DIM = 300
# WARM_STEPS = 4000

batch_size = 128
max_len = 64
d_model = 512
d_ff = 2048
n_head = 8
n_layer = 6
dropout = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warm_steps = 4000
pad_idx = 0
eps = 1e-6
src_vocab_size = 106107
trg_vocab_size = 22532
epochs = 20
clip = 1