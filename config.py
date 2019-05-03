# contains all configs for running main.py
import torch


class Config:
    # TODO
    batch_size = 128
    num_workers = 2
    z_dim = 100
    gen_features = 128
    gen_use_bias = False
    disc_features = 128
    disc_use_bias = False
    adam_beta1 = 0.5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
