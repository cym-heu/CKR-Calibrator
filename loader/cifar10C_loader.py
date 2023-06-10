import numpy as np
import os
import random

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

import config

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, name :str, root,
                 transform=None, target_transform=None):
        
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)


def load_cifar10C(cname):
    data_config = config.get_data_config('cifar10C')
    root = data_config['data_dir']
    corruptions = data_config['corruptions']
    assert cname in corruptions

    MEAN = [0.49139968, 0.48215841, 0.44653091]
    STD  = [0.24703223, 0.24348513, 0.26158784]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    dataset = CIFAR10C(cname, root, transform=transform)

    dataLoader = DataLoader(dataset, batch_size=data_config['batch_size'],
                            shuffle=False, num_workers=4)
    
    return dataLoader



