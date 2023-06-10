import numpy as np
import os
import random

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

import config


class MNISTC(datasets.VisionDataset):
    def __init__(self,  name :str, root,
                 transform=None, target_transform=None):
        super(MNISTC, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name, 'test_images.npy')
        target_path = os.path.join(root, name, 'test_labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img.squeeze(2))
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
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


def load_mnistC(cname):
    data_config = config.get_data_config('mnistC')
    root = data_config['data_dir']
    corruptions = data_config['corruptions']
    assert cname in corruptions

    transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])

    dataset = MNISTC(cname, root, transform=transform)

    dataLoader = DataLoader(dataset, batch_size=data_config['batch_size'],
                            shuffle=False, num_workers=4)
    
    return dataLoader