'''
给卷积核输出增加噪声，测试模型的准确率
'''
import torch
import sys
import os
sys.path.append('/home/gjs/code/robust_eval')

import loader
import models
import config
import util
from tqdm import tqdm
from core.modules_noise import HookModule_noise
from torch import nn
import time
from util import AverageStorage, ProgressMeter, accuracy, ClassAccuracy

# 超参
data_name = 'cifar10'
model_name = 'squeezenet'
SOTA = 'basic'
device_str = 'cuda:0'
layers = [1]

def test_noise(test_loader, model, criterion, device, num_classes, channel_dir):
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

        hookModule = HookModule_noise(model, layers, labels, channel_dir, device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        classAccuracy.accuracy(outputs, labels)

        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.item(), labels.size(0))

        progress.display(i)

        hookModule.remove_hook()

    return loss_meter, acc_meter, classAccuracy


def main():
    # device
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # 获取路径数据
    basic_dir = util.get_basic_dir(model_name=model_name, dataset_name=data_name, phase=SOTA)
    model_path = os.path.join(basic_dir, 'model_ori.pth')

    # 加载数据集
    train_loader = loader.load_data(data_name, 'train', SOTA=SOTA)
    test_loader = loader.load_data(data_name, 'test', SOTA=SOTA)
    data_config = config.get_data_config(data_name)
    in_channels, num_classes = data_config['in_channels'], data_config['num_classes']

    # 加载模型
    model = models.load_model(model_name, in_channels=in_channels, num_classes=num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # channel
    channel_dir = os.path.join(basic_dir, 'channels')

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    loss, acc, classAccuracy = test_noise(test_loader, model, criterion, device, num_classes, channel_dir)
    
    print('-' * 20 + 'Clean Dataset Test' + '-' * 20)
    print(loss, acc)
    print(classAccuracy)
    print('TIME CONSUMED', time.time() - since)


if __name__ == '__main__':
    main()