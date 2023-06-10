from torch.utils.data import DataLoader
from torchvision import transforms
from loader.datasets import ImageDataset
import config


def _get_train_set(data_path):
    return ImageDataset(image_dir=data_path,
                        mode='L',
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))


def _get_test_set(data_path):
    return ImageDataset(image_dir=data_path,
                        mode='L',
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))


def load_mnist(data_type=None):
    assert data_type is None or data_type in ['train', 'test']

    data_config = config.get_data_config('mnist')
    test_dir = data_config['test_dir']
    train_dir = data_config['train_dir']
    batch_size = data_config['batch_size']

    if data_type == 'train':
        data_set = _get_train_set(train_dir)
    else:
        data_set = _get_test_set(test_dir)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=True)

    return data_loader
