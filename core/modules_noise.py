import sys
sys.path.append('/home/gjs/code/robust_eval')
import torch
import os
from util import load_modules
import torch.nn.functional as F
import numpy as np


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
    # 选择指定卷积核增加噪声
    noise_std = 0.5
    # matrix_sel = matrix[mask]
    # noise = torch.normal(mean=0, std=noise_std, size=matrix_sel.shape).to(matrix.device)
    # matrix[mask] += noise

    # # 随机选择 1/10 的卷积核增加噪声
    k_num = matrix.shape[1] // 10
    random_idx = np.random.choice(matrix.shape[1], k_num, replace=False).tolist()
    noise = torch.normal(mean=0, std=noise_std, size=matrix[:, random_idx].shape).to(matrix.device)
    matrix[:, random_idx] -= F.relu(noise)
    return matrix

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
        self.robust_path = os.path.join(channel_dir, 'key_{}.pt')
        for idx, module in enumerate(modules):
            mask_pattern = torch.load(self.robust_path.format(layers[idx])).to(self.device)
            mask = (mask_pattern == 1)[labels]
            # 正常的中间层表征输出
            hookModule = HookModule_forward_noise(module=module, mask=mask, layer=layers[idx])
            self.hookModules.append(hookModule)

    def remove_hook(self):
        for hookModule in self.hookModules:
            hookModule.remove_hook()
