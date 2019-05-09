# defines training procedure
# for training GAN, some of advice from https://github.com/soumith/ganhacks was applied

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from utils import generate_new_image


class Trainer:
    """
    Trainer for DCGAN.
    """
    def __init__(self, generator, discriminator, dataloader, use_stochastic_soft_label=True):
        """
        :param generator: Generator network from model.py
        :param discriminator: Discriminator network from model.py
        :param dataloader: Data Loader from dataloader.py
        :param use_stochastic_soft_label: if True, then for discriminator, labels for true images are drawn from [0.7, 1) and labels for fake iamges are drawn from [0, 0.3) following uniform distribution. If False, then labels for true images are 1, and labels for fake images are 0, deterministically.
        """
        self.generator = generator.to(Config.device)
        self.discriminator = discriminator.to(Config.device)
        self.dataloader = dataloader

        # disc_optimizer is SGD, following advice from https://github.com/soumith/ganhacks
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=Config.lr, betas=(Config.adam_beta1, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=Config.lr, betas=(Config.adam_beta1, 0.999))

        self.criterion = nn.BCELoss().to(Config.device)

        self.gen_loss_hist = []
        self.disc_loss_hist = []

        self.curr_epoch = 0

        # to avoid discriminator overfitting
        self.stochastic_soft_label = use_stochastic_soft_label

    def train(self, num_epochs):
        fixed_z = torch.rand((8*8, 100, 1, 1), device=Config.device)
        dataloader_len = len(self.dataloader)

        for epoch in range(self.curr_epoch, num_epochs):
            epoch_loss_D = 0
            epoch_loss_G = 0

            start = time.time()

            for ix, data in enumerate(self.dataloader, 0):
                images = data[0].to(Config.device)
                loss_D, loss_G = self.train_step(images)
                epoch_loss_D += loss_D
                epoch_loss_G += loss_G

                # print every 100 iteration = 12800 (100 * batch_size) images seen
                if (ix % 100) == 0:
                    print('[{0}/{1}] [{2}/{3}] loss_D: {4:.4f}, loss_G: {5:.4f}'.format(epoch + 1, num_epochs,
                                                                                ix + 1, dataloader_len,
                                                                                loss_D, loss_G))

            print('epoch {0} {1} seconds loss_D: {2:.4f} loss_G: {3:.4f}'.format(epoch + 1, time.time() - start,
                                                                                    epoch_loss_D / dataloader_len,
                                                                                    epoch_loss_G / dataloader_len
                                                                                    ))

            # at the end of each epoch, generate image and save
            if not os.path.isdir('images/{0}/'.format(self.dataloader.name)):
                if not os.path.isdir('images/'):
                    os.mkdir('images')
                os.mkdir('images/{0}/'.format(self.dataloader.name))
            generate_new_image(self.generator, z=fixed_z, show=False, save_path='images/{0}/epoch-{1}.jpg'.format(
                                                                                                self.dataloader.name,
                                                                                                self.curr_epoch + 1))
            self.curr_epoch += 1

        if not os.path.isdir('checkpoints/'):
            os.mkdir('checkpoints/')
        self.save_checkpoint('checkpoints/{0}-epoch-{1}.ckpt'.format(self.dataloader.name,
                                                                     num_epochs))

        return self.disc_loss_hist, self.gen_loss_hist

    def train_step(self, images):
        loss_D = 0
        loss_G = 0

        batch_size = images.shape[0]  # may not equal to our target batch_size in last batch

        # 1. First, train Discriminator
        self.discriminator.zero_grad()

        # 1-1. train with real images
        images_disc_output = self.discriminator(images)

        if self.stochastic_soft_label:
            real_target = torch.rand_like(images_disc_output) * 0.3 + 0.7  # between 0.7 and 1
        else:
            real_target = torch.ones_like(images_disc_output)

        loss_real = self.criterion(images_disc_output, real_target)
        loss_real.backward()
        loss_D += loss_real.item()

        # 1-2. next, train Discriminator with fake images
        # Z follows uniform distribution, as suggested in DCGAN paper
        random_z = torch.rand((batch_size, 100, 1, 1), device=Config.device)
        fake_images = self.generator(random_z)

        fake_images_disc_output = self.discriminator(fake_images)

        if self.stochastic_soft_label:
            fake_target = torch.rand_like(fake_images_disc_output) * 0.3  # between 0 and 0.3
        else:
            fake_target = torch.zeros_like(fake_images_disc_output)

        loss_fake = self.criterion(fake_images_disc_output, fake_target)
        loss_fake.backward()
        loss_D += loss_fake.item()

        self.disc_optimizer.step()

        # 2. Train Generator
        self.generator.zero_grad()

        random_z = torch.rand((batch_size, 100, 1, 1), device=Config.device)
        fake_images = self.generator(random_z)
        fake_images_disc_output = self.discriminator(fake_images)

        # try alternative generator loss, to prevent vanishing gradient in early stage
        # we want to maximize log D, instead of minimizing log (1-D)
        # <=> equivalent to minmizing -log D
        loss = -1 * torch.log(fake_images_disc_output)
        loss.backward()
        loss_G += loss.item()

        # append loss to history
        self.disc_loss_hist.append(loss_D)
        self.gen_loss_hist.append(loss_G)

        return loss_D, loss_G

    def grad_reset(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

    def save_checkpoint(self, checkpoint_path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'epoch': self.curr_epoch,
            'gen_loss_hist': self.gen_loss_hist,
            'disc_loss_hist': self.disc_loss_hist
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.curr_epoch = checkpoint['epoch'] + 1
        self.gen_loss_hist = checkpoint['gen_loss_hist']
        self.disc_loss_hist = checkpoint['disc_loss_hist']

