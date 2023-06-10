'''
寻找关键的卷积核
'''
import torch
import sys
import os
sys.path.append('/home/gjs/code/robust_eval')

import loader
import models
# todo
import config
import util
from tqdm import tqdm
from core.modules_key import HookModule_key

# 超参
data_name = 'cifar10'
model_name = 'squeezenet'
SOTA = 'basic'
device_str = 'cuda:1'
layers = [1]
corruption = False

def screen(model, train_loader, num_classes, device, save_dir):
    hookModule = HookModule_key(model, layers, num_classes)
    with tqdm(total=len(train_loader), ncols=80)as pbar:
        for idx, samples in tqdm(enumerate(train_loader)):
            inputs, labels = samples
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            hookModule.collect(outputs, labels)

            # pbar.set_postfix_str({'matrix_shape: {}'.format(hookModule.matrix.shape)})
            pbar.update()
    hookModule.find_kernel(save_dir)
    hookModule.remove_hook()


def main():
    # device
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # 获取路径数据
    basic_dir = util.get_basic_dir(model_name=model_name, dataset_name=data_name, phase=SOTA)
    model_path = os.path.join(basic_dir, 'model_ori.pth')
    save_dir = os.path.join(basic_dir, 'channels')

    # 加载数据集
    train_loader = loader.load_data(data_name, 'train', SOTA=SOTA)
    data_config = config.get_data_config(data_name)
    in_channels, num_classes = data_config['in_channels'], data_config['num_classes']

    # 加载模型
    model = models.load_model(model_name, in_channels=in_channels, num_classes=num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 寻找关键的卷积核
    screen(model, train_loader, num_classes, device, save_dir)


if __name__ == '__main__':
    main()