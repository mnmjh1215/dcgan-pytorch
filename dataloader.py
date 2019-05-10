# data loader for part of LSUN bedroom (around 10% of the entire data set),
# STL-10 (instead of Imagenet-1k data set used in DCGAN paper)
# and CelebA (instead of face data set used and created by DCGAN authors)

import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# transform to be used for most(?) data sets
common_transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


def load_STL10(batch_size=128, num_workers=2):
    """

    :param batch_size: batch_size for torch.utils.data.DataLoader
    :param num_workers: num_workers for torch.utils.data.DataLoader
    :return: data loader for STL10 data set
    """
    # check if root directory exists
    root = 'data/STL10/'
    if not os.path.isdir(root):
        if not os.path.isdir('data'):
            os.mkdir('data')
        os.mkdir(root)

    STL10_dataset = datasets.STL10(
        root=root,
        split="train+unlabeled",
        transform=common_transform,
        download=True
    )

    dataloader = DataLoader(STL10_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers)
    dataloader.name = 'STL10'

    return dataloader


def load_CelebA(batch_size=128, num_workers=2, download=False):
    """
    load CelebA data set, download if download is True and Kaggle API is available

    :param batch_size: batch_size for torch.utils.data.DataLoader
    :param num_workers: num_workers for torch.utils.data.DataLoader
    :param download: whether to download data set. Kaggle API is required
    :return: data loader for CelebA data set
    """

    # Couldn't figure out the best way to download data
    # because the original link in official CelebA website is not available
    # recommend downloading from following url, or use Kaggle API.
    # https://www.kaggle.com/jessicali9530/celeba-dataset/downloads/celeba-dataset.zip/2

    # check if root directory exists
    root = 'data/CelebA/'
    if not os.path.isdir(root):
        if not os.path.isdir('data'):
            os.mkdir('data')
        os.mkdir(root)

    if download and not os.path.isdir('data/CelebA/img_align_celeba'):
        import subprocess
        subprocess.run(['kaggle', 'datasets', 'download', 'jessicali9530/celeba-dataset', '-p', 'data/CelebA'])
        subprocess.run(['unzip', 'data/CelebA/celeba-dataset.zip', '-d', 'data/CelebA'])
        subprocess.run(['unzip', 'data/CelebA/img_align_celeba.zip', '-d', 'data/CelebA'])
        subprocess.run(['rm', 'data/CelebA/celeba-dataset.zip', 'data/CelebA/img_align_celeba.zip'])

    CelebA_dataset = datasets.ImageFolder(
        root=root,
        transform=common_transform
    )

    dataloader = DataLoader(CelebA_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers)
    dataloader.name = 'CelebA'

    return dataloader


def load_LSUN(batch_size=128, num_workers=2, download=False):
    """
    load about 10% of LSUN bedroom data set, download if download is True and Kaggle API is available

    :param batch_size: batch_size for torch.utils.data.DataLoader
    :param num_workers: num_workers for torch.utils.data.DataLoader
    :param download: whether to download data set. Kaggle API is required
    :return: data loader for LSUN data set
    """

    # Since LSUN bedroom data is too large, we will use only 10% of them
    # https://www.kaggle.com/jhoward/lsun_bedroom

    # check if root directory exists
    root = 'data/LSUN/'
    if not os.path.isdir(root):
        if not os.path.isdir('data'):
            os.mkdir('data')
        os.mkdir(root)

    if download and not os.path.isdir('data/LSUN/sample'):
        import subprocess
        subprocess.run(['kaggle', 'datasets', 'download', 'jhoward/lsun_bedroom', '-p', 'data/LSUN'])
        subprocess.run(['unzip', 'data/LSUN/lsun_bedroom.zip'])
        subprocess.run(['unzip', 'data/LSUN/sample.zip'])
        subprocess.run(['rm', 'data/LSUN/lsun_bedroom.zip', 'data/LSUN/sample.zip'])

    LSUN_dataset = datasets.ImageFolder(
        root=root,
        transform=common_transform
    )

    dataloader = DataLoader(LSUN_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers)
    dataloader.name = 'LSUN'

    return dataloader


