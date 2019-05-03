# main function for training and testing

import torch
from model import Generator, Discriminator
from config import Config
from dataloader import *
from train import Trainer
from utils import load_model, generate_new_image, get_args


def main():
    device = Config.device

    generator = Generator(z_dim=Config.z_dim, gen_features=Config.gen_features, use_bias=Config.gen_use_bias).to(device)
    discriminator = Discriminator(disc_features=Config.disc_features, use_bias=Config.disc_use_bias).to(device)

    args = get_args()

    if args.model_path:
        load_model(generator, discriminator, args.model_path)

    if args.test:
        # Do some test stuff
        print("testing...")
        generate_new_image(generator, show=True, save_path=args.image_save_path)

    else:
        # not testing means... training!
        print('training with {0}...'.format(args.dataset))
        if args.dataset == 'CelebA':
            dataloader = load_CelebA(Config.batch_size, Config.num_workers)
        elif args.dataset == 'STL10':
            dataloader = load_STL10(Config.batch_size, Config.num_workers)
        elif args.dataset == 'LSUN':
            dataloader = load_LSUN(Config.batch_size, Config.num_workers)

        print("loading trainer...")
        trainer = Trainer(generator, discriminator, dataloader)
        print("start training...")
        trainer.train(Config.num_epochs)


if __name__ == '__main__':
    main()
