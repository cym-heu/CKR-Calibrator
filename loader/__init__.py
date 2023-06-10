from loader.cifar10_loader import load_cifar10, load_cifar10_augmix, load_cifar10_cutout, load_cifar10_autoaug
from loader.cifar10C_loader import load_cifar10C

from loader.cifar100_loader import load_cifar100, load_cifar100_augmix
from loader.cifar100C_loader import load_cifar100C

from loader.mnist_loader import load_mnist
from loader.mnistC_loader import load_mnistC

def load_data(data_name, data_type='train', SOTA='None'):
    print('-' * 50)
    print('DATA NAME:', data_name)
    print('DATA TYPE:', data_type)
    print('-' * 50)
    
    assert data_name in ['cifar10', 'cifar100', 'mnist', 'tiny']
    if data_name == 'cifar10':
        data_loader = load_cifar10(data_type)
        if SOTA == 'augmix':
            data_loader = load_cifar10_augmix(data_type)
        if SOTA == 'cutout':
            data_loader = load_cifar10_cutout(data_type)
        if SOTA == 'autoaug':
            data_loader = load_cifar10_autoaug(data_type)
    if data_name == 'cifar100':
        data_loader = load_cifar100(data_type)
        if SOTA == 'augmix':
            data_loader = load_cifar100_augmix(data_type)
    if data_name == 'mnist':
        data_loader = load_mnist(data_type)
    
    return data_loader

def load_data_C(data_name, type_c):

    assert data_name in ['cifar10C', 'cifar100C', 'mnistC']
    if data_name == 'cifar10C':
        data_loader = load_cifar10C(type_c)
    if data_name == 'cifar100C':
        data_loader = load_cifar100C(type_c)
    if data_name == 'mnistC':
        data_loader = load_mnistC(type_c)
    return data_loader