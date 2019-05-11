# utils
import torch
import torch.nn as nn
import argparse
from config import Config
import torchvision.utils as tvutils
import matplotlib.pyplot as plt
import numpy as np


def generate_new_image(generator, z=None, show=False, save_path=None):
    if z is None:
        z = torch.randn((8*8, 100, 1, 1), device=Config.device)

    generator.eval()
    new_images = generator(z).detach().cpu()
    generator.train()

    image_grid = tvutils.make_grid(new_images, nrow=8, padding=2, normalize=True, range=(-1, 1))

    if show:
        plt.imshow(np.transpose(image_grid, (1, 2, 0)))
        plt.show()

    if save_path:
        tvutils.save_image(new_images, save_path, nrow=8, padding=2, normalize=True, range=(-1, 1))

    return image_grid


def load_model(generator, discriminator, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        default='CelebA',
                        const='CelebA',
                        nargs='?',
                        choices=['LSUN', 'CelebA', 'STL10'],
                        help='Data set to train your model')

    parser.add_argument('--test',
                        action='store_true',
                        help='use this argument to test generator (= create image)')

    parser.add_argument('--model_path',
                        help='path to saved model, required for testing and resuming training. '
                             'If not given in training, model will be trained from scratch with given data set')

    parser.add_argument('--image_save_path',
                        help='path to save generated image, required to save image generated at testing')

    parser.add_argument('--download_dataset',
                        default=False,
                        action='store_true',
                        help='use this argument to download dataset')

    args = parser.parse_args()

    return args


