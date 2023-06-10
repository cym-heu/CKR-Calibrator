import torch
import argparse
import util
import os
import time
import pprint

from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import loader
import models
import config
from util import AverageStorage, ProgressMeter, accuracy, ClassAccuracy

def test_c(data_loader, model, criterion, device, num_classes):
    loss_meter = AverageStorage('Loss', ':4e')
    acc_meter = AverageStorage('Acc', ':.2%')
    model.eval()

    for i, samples in enumerate(data_loader):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.item(), labels.size(0))


    return loss_meter, acc_meter

def test(test_loader, model, criterion, device, num_classes):
    loss_meter = AverageStorage('Loss', ':4e')
    acc_meter = AverageStorage('Acc', ':.2%')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Testing',
                             meters=[loss_meter, acc_meter])
    classAccuracy = ClassAccuracy(num_classes)
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        classAccuracy.accuracy(outputs, labels)

        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.item(), labels.size(0))

        progress.display(i)

    return loss_meter, acc_meter, classAccuracy

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--SOTA', type=str, default='basic')
    parser.add_argument('--model_type', type=str, default='optim')
    parser.add_argument('--corruption', action='store_true', default=False)
    parser.add_argument('--layers', type=int, default='1', nargs='+')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 模型、日志保存路径
    basic_dir = util.get_basic_dir(args.model_name, args.dataset_name, args.SOTA)
    log_dir = os.path.join(basic_dir, 'runs')
    model_path = os.path.join(basic_dir, 'model_' + args.model_type + '.pth')
    if args.model_type == 'optim' and args.layers[0] != 1:
        model_path = os.path.join(basic_dir, 'model_' + args.model_type + '_{}.pth'.format(args.layers[0]))

    # 数据集
    test_loader = loader.load_data(args.dataset_name, data_type='test')
    data_config = config.get_data_config(args.dataset_name)

    # 模型
    in_channels, num_classes = data_config['in_channels'], data_config['num_classes']
    model = models.load_model(args.model_name, in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir)


    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    loss, acc, classAccuracy = test(test_loader, model, criterion, device, num_classes)
    
    print('-' * 20 + 'Clean Dataset Test' + '-' * 20)
    print(loss, acc)
    print(classAccuracy)
    print('TIME CONSUMED', time.time() - since)

    since = time.time()

    if args.corruption:
        data_name = args.dataset_name + 'C'
        data_config = config.get_data_config(data_name)
        corruptions = data_config['corruptions']

        accs = {}
        with tqdm(total=len(corruptions), ncols=80) as pbar:
            for idx, cname in enumerate(corruptions):
                data_loader = loader.load_data_C(data_name, cname)
                loss, acc = test_c(data_loader, model, criterion, device, num_classes)
                accs[cname] = acc.avg
                pbar.set_postfix_str(f'{cname}: {acc.avg:.2f}')
                pbar.update()
        avg = np.mean(list(accs.values()))
        accs['avg'] = avg
        print('-' * 20 + 'Corruption Dataset Test' + '-' * 20)
        pprint.pprint(accs)
        print('TIME CONSUMED', time.time() - since)



if __name__ == '__main__':
    main()