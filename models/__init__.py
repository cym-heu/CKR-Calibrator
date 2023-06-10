import torch
import torch.nn as nn
from models import wideresnet, resnet, squeezenet, alexnet, vgg, resnext, densenet

def load_model(model_name, in_channels=3, num_classes=10):
    print('-' * 50)
    print('LOAD MODEL:', model_name)
    print('-' * 50)

    if model_name == 'wideresnet':
        model = wideresnet.wideresnet(in_channels, num_classes)
    elif model_name == 'resnet50':
        model = resnet.resnet50(in_channels, num_classes)
    elif model_name == 'squeezenet':
        model = squeezenet.squeezenet(in_channels, num_classes)
    elif model_name == 'alexnet':
        model = alexnet.alexnet(in_channels, num_classes)
    elif model_name == 'resnext':
        model = resnext.resnext50(in_channels, num_classes)
    elif model_name == 'vgg16':
        model = vgg.vgg16_bn(in_channels, num_classes)
    elif model_name == 'densenet':
        model = densenet.densenet121(in_channels, num_classes)
    return model