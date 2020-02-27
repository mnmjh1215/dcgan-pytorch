# data loader for part of LSUN bedroom (around 10% of the entire data set),
# STL-10 (instead of Imagenet-1k data set used in DCGAN paper)
# and CelebA (instead of face data set used and created by DCGAN authors)

import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# transform to be used for most(?) data sets
default_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


def load_STL10(batch_size=64, num_workers=2, download=False):
    # check if root directory exists
    root = 'data/STL10/'
    if not os.path.isdir(root):
        os.makedirs(root)

    STL10_dataset = datasets.STL10(
        root=root,
        split="train+unlabeled",
        transform=default_transforms,
        download=download
    )

    dataloader = DataLoader(STL10_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=True)
    dataloader.name = 'STL10'

    return dataloader


def load_CelebA(batch_size=64, num_workers=2, download=False):

    # recommend downloading from following url, or use Kaggle API.
    # https://www.kaggle.com/jessicali9530/celeba-dataset/data

    # check if root directory exists
    root = 'data/CelebA/'
    if not os.path.isdir(root):
        os.makedirs(root)

    if download and not os.path.isdir('data/CelebA/img_align_celeba'):
        import subprocess
        subprocess.run(['kaggle', 'datasets', 'download', 'jessicali9530/celeba-dataset', '-p', 'data/CelebA'])
        subprocess.run(['unzip', '-q', 'data/CelebA/celeba-dataset.zip', '-d', 'data/CelebA'])
        subprocess.run(['mv', 'data/CelebA/img_align_celeba/img_align_celeba/*', 'data/CelebA/img_align_celeba'])
        subprocess.run(['rm', 'data/CelebA/celeba-dataset.zip'])
        subprocess.run(['rmdir', 'data/CelebA/img_align_celeba/img_align_celeba'])

    CelebA_dataset = datasets.ImageFolder(
        root=root,
        transform=default_transforms
    )

    dataloader = DataLoader(CelebA_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=True)
    dataloader.name = 'CelebA'

    return dataloader


def load_LSUN(batch_size=128, num_workers=2, download=False):

    # Since LSUN bedroom data is too large, we will use only 10% of them
    # https://www.kaggle.com/jhoward/lsun_bedroom

    # check if root directory exists
    root = 'data/LSUN/'
    if not os.path.isdir(root):
        os.makedirs(root)

    if download and not os.path.isdir('data/LSUN/sample'):
        import subprocess
        subprocess.run(['kaggle', 'datasets', 'download', 'jhoward/lsun_bedroom', '-p', 'data/LSUN'])
        subprocess.run(['unzip', '-q', 'data/LSUN/lsun_bedroom.zip', '-d', 'data/LSUN'])
        subprocess.run(['unzip', '-q', 'data/LSUN/sample.zip', '-d', 'data/LSUN'])
        subprocess.run(['rm', 'data/LSUN/lsun_bedroom.zip', 'data/LSUN/sample.zip'])

    LSUN_dataset = datasets.ImageFolder(
        root=root,
        transform=default_transforms
    )

    dataloader = DataLoader(LSUN_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=True)
    dataloader.name = 'LSUN'

    return dataloader


