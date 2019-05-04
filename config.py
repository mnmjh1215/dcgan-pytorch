# contains all configs for running main.py
import torch


class Config:
    # dataloader.py
    batch_size = 128
    num_workers = 2
    # model.py
    z_dim = 100
    gen_features = 128
    gen_use_bias = True
    disc_features = 128
    disc_use_bias = True
    # train.py
    adam_lr = 0.0002
    adam_beta1 = 0.5
    # main.py
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 20

