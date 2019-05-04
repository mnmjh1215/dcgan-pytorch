# dcgan model implemented in pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    # initialize convolution and transposed convolution weights with mean 0.0 and standard deviation 0.02
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


class Generator(nn.Module):
    """
    Generator network in DCGAN.
    Implemented to strictly follow original DCGAN paper's description as much as possible.
    """
    def __init__(self, z_dim=100, gen_features=128, use_bias=True):
        """
        :param z_dim: dimension of random vector z (default is 100, which is used in DCGAN paper for LSUN dataset)
        :param gen_features: features (or channels) of last layer before generating image.
        :param use_bias: since authors of DCGAN did not specify whether they used bias on transposed convolution layer,
                         it is left as an option.
        """

        super().__init__()
        self.z_dim = z_dim

        self.conv_transpose1 = nn.ConvTranspose2d(self.z_dim, gen_features * 8, 4, 1, 0, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(gen_features * 8)
        # (gen_features * 8, 4, 4)

        self.conv_transpose2 = nn.ConvTranspose2d(gen_features * 8, gen_features * 4, 4, 2, 1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(gen_features * 4)
        # (gen_features * 4, 8, 8)

        self.conv_transpose3 = nn.ConvTranspose2d(gen_features * 4, gen_features * 2, 4, 2, 1, bias=use_bias)
        self.bn3 = nn.BatchNorm2d(gen_features * 2)
        # (gen_features * 2, 16, 16)

        self.conv_transpose4 = nn.ConvTranspose2d(gen_features * 2, gen_features, 4, 2, 1, bias=use_bias)
        self.bn4 = nn.BatchNorm2d(gen_features)
        # (gen_features, 32, 32)

        self.conv_transpose5 = nn.ConvTranspose2d(gen_features, 3, 4, 2, 1, bias=use_bias)
        self.tanh = nn.Tanh()
        # (3, 64, 64) is our target image size

        # initialize weights according to DCGAN paper
        self.apply(init_weights)

    def forward(self, input):
        """

        :param input: uniform noise vector (or vectors) with shape (batch_size, z_dim, 1, 1)
        :return: images generated by the generator, with shape (batch_size, 3, 64, 64)
        """
        x = F.relu(self.bn1(self.conv_transpose1(input)))
        x = F.relu(self.bn2(self.conv_transpose2(x)))
        x = F.relu(self.bn3(self.conv_transpose3(x)))
        x = F.relu(self.bn4(self.conv_transpose4(x)))
        x = self.tanh(self.conv_transpose5(x))  # nn.functional.tanh is deprecated

        return x


class Discriminator(nn.Module):
    """
    Discriminator model in DCGAN
    """

    def __init__(self, disc_features=128, use_bias=False):
        """

        :param disc_features: features (or channels) of first convolution layer
        :param use_bias: since authors of DCGAN did not specify whether they used bias on convolution layer,
                         it is left as an option.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, disc_features, 4, 2, 1)
        # (disc_features, 32, 32)

        self.conv2 = nn.Conv2d(disc_features, disc_features * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(disc_features * 2)
        # (disc_features * 2, 16, 16)

        self.conv3 = nn.Conv2d(disc_features * 2, disc_features * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(disc_features * 4)
        # (disc_features * 4, 8, 8)

        self.conv4 = nn.Conv2d(disc_features * 4, disc_features * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(disc_features * 8)
        # (disc_features * 8, 4, 4)

        self.conv5 = nn.Conv2d(disc_features * 8, 1, 4, 1, 0)
        self.sigmoid = nn.Sigmoid()
        # (1, 1, 1)

        self.apply(init_weights)

    def forward(self, input):
        """

        :param input: images to be discriminated, with shape (batch_size, 3, 64, 64)
        :return: sigmoid output, with shape (batch_size, 1)
        """
        x = F.leaky_relu(self.conv1(input), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = self.sigmoid(self.conv5(x))
        x = x.view(-1)  # flattened, so that it can be directly used to calculate loss

        return x
