from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loader.datasets import ImageDataset
from loader.datasets.augmix import AugMixDataset
import config

import torch


def _get_train_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            # transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                        ]))


def _get_test_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                        ]))

def load_cifar100(data_type=None):
    assert data_type is None or data_type in ['train', 'test']

    data_config = config.get_data_config('cifar100')
    test_dir = data_config['test_dir']
    train_dir = data_config['train_dir']
    batch_size = data_config['batch_size']
    train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                    ])
    test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                    ])
    if data_type == 'train':
        dataset = datasets.CIFAR100(
            root='/home/gjs/code/md_up/SOTA/crossnorm_selfnorm_main/data/', train=True, transform=train_transform, download=False)
    else:
        dataset = datasets.CIFAR100(
            root='/home/gjs/code/md_up/SOTA/crossnorm_selfnorm_main/data/', train=False, transform=test_transform, download=False)
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)
    return data_loader

# def load_cifar100(data_type=None):
#     assert data_type is None or data_type in ['train', 'test']

#     data_config = config.get_data_config('cifar100')
#     test_dir = data_config['test_dir']
#     train_dir = data_config['train_dir']
#     batch_size = data_config['batch_size']


#     if data_type == 'train':
#         data_set = _get_train_set(train_dir)
#     else:
#         data_set = _get_test_set(test_dir)

#     data_loader = DataLoader(dataset=data_set,
#                              batch_size=batch_size,
#                              num_workers=4,
#                              shuffle=True)

#     return data_loader


def load_cifar100_augmix(data_type=None):
    assert data_type is None or data_type in ['train', 'test']

    data_config = config.get_data_config('cifar100')
    test_dir = data_config['test_dir']
    train_dir = data_config['train_dir']
    batch_size = data_config['batch_size']

    if data_type == 'train':
        data_set = ImageDataset(image_dir=train_dir,
                                transform=transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                ]))
        preprocess = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))])
        data_set = AugMixDataset(data_set, preprocess, all_ops=False, mixture_width=3,
                                   mixture_depth=-1, aug_severity=3, no_jsd=False, image_size=32)

    else:
        data_set = ImageDataset(image_dir=test_dir,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))
                                ]))
    


    data_loader = DataLoader(dataset=data_set,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=True)

    return data_loader