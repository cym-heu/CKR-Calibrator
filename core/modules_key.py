import sys
sys.path.append('/home/gjs/code/robust_eval')
import torch
import numpy as np
from tqdm import tqdm
import os
from util import load_modules

 
class HookModule_grad:
    def __init__(self, module, layer):
        self.inputs = None
        self.outputs = None
        self.layer = layer
        self.hook = module.register_forward_hook(self._hook)

    def grads(self, outputs, inputs=None, retain_graph=True, create_graph=False):
        if inputs is None:
            inputs = self.outputs  # default the output dim

        return torch.autograd.grad(outputs=outputs,
                                   inputs=inputs,
                                   retain_graph=retain_graph,
                                   create_graph=create_graph)[0]

    def _hook(self, module, inputs, outputs):
        self.inputs = inputs[0]
        self.outputs = outputs

    def remove_hook(self):
        self.hook.remove()


def _normalization(data, axis=None, bot=False):
    assert axis in [None, 0, 1]
    _max = np.max(data, axis=axis)
    if bot:
        _min = np.zeros(_max.shape)
    else:
        _min = np.min(data, axis=axis)
    _range = _max - _min
    if axis == 1:
        _norm = ((data.T - _min) / (_range + 1e-5)).T
    else:
        _norm = (data - _min) / (_range + 1e-5)
    return _norm

class HookModule_key:
    '''
    寻找关键卷积核
    1. 在执行model(x)之前注册HookModule(for循环外)
    2. 在执行model(x)之后调用collect函数收集卷积核得分(for循环内)
    3. 在模型运算结束后调用sift函数处理和保存重要卷积核(for循环外)
    4. 在模型运算结束后调用函数remove_hook(for循环外)
    '''
    def __init__(self, model, layers, num_classes):
        modules = load_modules(model, layers)
        self.hookModules = []
        self.threshold = 0.2
        self.values = [[[] for _ in range(num_classes)] for _ in range(len(layers))]
        self.layers = layers
        for idx, module in enumerate(modules):
            hookModule = HookModule_grad(module=module, layer=layers[idx])
            self.hookModules.append(hookModule)

    def _normalization(self, data, axis=None, bot=False):
        assert axis in [None, 0, 1]
        _max = np.max(data, axis=axis)
        if bot:
            _min = np.zeros(_max.shape)
        else:
            _min = np.min(data, axis=axis)
        _range = _max - _min
        if axis == 1:
            _norm = ((data.T - _min) / (_range + 1e-5)).T
        else:
            _norm = (data - _min) / (_range + 1e-5)
        return _norm
    
    def collect(self, outputs, labels):
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        for layer, hookModule in enumerate(self.hookModules):

            values = hookModule.grads(-nll_loss, hookModule.outputs)
            values = torch.relu(values)

            values = values.detach().cpu().numpy()

            for b in range(len(labels)):
                self.values[layer][labels[b]].append(values[b])
    

    def find_kernel(self, result_path):
        for idx, values in enumerate(tqdm(self.values)):
            layer = self.layers[idx]
            values = np.asarray(values)
            if len(values.shape) > 3:
                values = np.sum(values, axis=(3, 4))  # [num_classes, num_images, channels]
            values = np.sum(values, axis=1)  # [num_classes, channels]
            values = _normalization(values, axis=1)

            mask = np.zeros(values.shape)
            mask[np.where(values > self.threshold)] = 1
            mask = torch.from_numpy(mask)
            mask_path = os.path.join(result_path, 'key_{}.pt'.format(layer))
            torch.save(mask, mask_path)

            print('-' * 50)
            print('Saving key mask in layer_{}'.format(layer))
            print('mask.shape', mask.shape)
            print('mask.sum', mask.sum())

    def remove_hook(self):
        for hookModule in self.hookModules:
            hookModule.remove_hook()


