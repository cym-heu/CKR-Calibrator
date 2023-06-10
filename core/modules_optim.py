# 模型优化的核心代码
import sys
sys.path.append('/home/gjs/code/robust_eval')
import torch
import numpy as np
from util import load_modules, batch_hard_triplet_loss
import os
import torch.nn.functional as F
import config


class HookModule_trip:
    '''
    为模型中间层添加噪声
    1. 在for循环前初始化
    2. 在for循环中（model()之后）调用optim
    2. 在for循环后romove_hook
    '''
    def __init__(self, model, layers, channel_dir, device):
        modules = load_modules(model, layers)
        self.hookModules = []
        self.device = device
        self.means_path = os.path.join(channel_dir, 'means_{}.pt')
        for idx, module in enumerate(modules):
            # 正常的中间层表征输出
            hookModule = HookModule_forward_trip(module=module, layer=layers[idx])
            self.hookModules.append(hookModule)
    
    def optim(self, labels):
        loss = 0.0
        for hookModule in self.hookModules:
            layer = hookModule.layer
            matrix = hookModule.matrix
            means = torch.load(self.means_path.format(layer)).to(self.device)
            # 表征拉近
            loss += self._triplet(matrix, means, labels)

        return loss

    def _triplet(self, matrix, means, labels):
        if len(matrix.shape) > 2:
            matrix = matrix.reshape(matrix.shape[0], matrix.shape[1], -1).mean(dim=-1)
        loss = batch_hard_triplet_loss(labels, matrix, margin=0.05) * config.beta
        return loss
        

    def remove_hook(self):
        for hookModule in self.hookModules:
            hookModule.remove_hook()

class HookModule_forward_trip:
    def __init__(self, module, layer):
        self.module = module
        self.matrix = None
        self.layer = layer
        self.hook = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.matrix = outputs
        return outputs
        
    def remove_hook(self):
        self.hook.remove()


class HookModule_noise:
    '''
    为模型中间层添加噪声
    1. 在for循环中初始化
    2. 在for循环中model()后romove_hook()
    '''
    def __init__(self, model, layers, labels, channel_dir, device):
        modules = load_modules(model, layers)
        self.hookModules = []
        self.device = device
        self.robust_path = os.path.join(channel_dir, 'robust_{}.pt')
        self.key_path = os.path.join(channel_dir, 'key_{}.pt')
        for idx, module in enumerate(modules):
            robust_mask_pattern = torch.load(self.robust_path.format(layers[idx])).to(self.device)
            robust_mask = (robust_mask_pattern == 1)[labels]
            key_mask_pattern = torch.load(self.key_path.format(layers[idx])).to(self.device)
            key_mask = (key_mask_pattern == 1)[labels]
            # 正常的中间层表征输出
            hookModule = HookModule_forward_noise(module=module, mask=key_mask, layer=layers[idx])
            self.hookModules.append(hookModule)

    def remove_hook(self):
        for hookModule in self.hookModules:
            hookModule.remove_hook()


class HookModule_forward_noise:
    def __init__(self, module, mask, layer):
        self.module = module
        self.matrix = None
        self.layer = layer
        self.mask = mask
        self.hook = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.matrix = outputs
        add_noise(outputs, self.mask)
        return outputs
        
    def remove_hook(self):
        self.hook.remove()


def add_noise(matrix, mask):
    noise_std = 0.5
    # 选择不够鲁棒的卷积核添加噪声
    matrix_sel = matrix[mask]
    noise = torch.normal(mean=0, std=noise_std, size=matrix_sel.shape).to(matrix.device)
    matrix[mask] -= F.relu(noise)

    # 随机选取1/10的卷积核添加噪声
    # k_num = matrix.shape[1] // 10
    # random_idx = np.random.choice(matrix.shape[1], k_num, replace=False).tolist()
    # noise = torch.normal(mean=0, std=noise_std, size=matrix[:, random_idx].shape).to(matrix.device)
    # matrix[:, random_idx] += noise