import sys
sys.path.append('/home/gjs/code/robust_eval')
import torch
import config
import os
from util import load_modules, get_basic_dir

from torch import nn



class HookModule_forward:
    def __init__(self, module, layer):
        self.module = module
        self.matrix = None
        self.layer = layer
        self.hook = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.matrix = outputs
        
    def remove_hook(self):
        self.hook.remove()


class HookModule:
    '''
    统计表征输出值
    1. 在for循环前初始化
    2. 在执行model(x)之后调用collect_函数收集表征信息(for循环内)
    3. 在模型运算结束后调用gaussian_dis函数统计均值和方差(for循环外)
    4. 在模型运算结束后调用函数remove_hook(for循环外)
    '''
    def __init__(self, model, layers, num_classes=10):
        modules = load_modules(model, layers)
        self.hookModules = []
        self.gaus_mean, self.gaus_std = None, None
        self.num_classes = num_classes
        self.features_all = {layer: [[] for c in range(num_classes)] for layer in layers} # 所有类别的表征
        self.alpha = config.alpha
        for idx, module in enumerate(modules):
            # 正常的中间层表征输出
            hookModule = HookModule_forward(module=module, layer=layers[idx])
            self.hookModules.append(hookModule)

    def _gen_features(self, detach=False):
        feature_dict = {}
        for hookModule in self.hookModules:
            feature = hookModule.matrix
            layer = hookModule.layer
            if len(feature.shape) > 2:
                feature = feature.reshape(feature.shape[0], feature.shape[1], -1).mean(dim=-1)
            if detach:
                feature = feature.detach().clone()
            feature_dict[layer] = feature
        return feature_dict
    
    def _norm(self, f):
        # 对B*C的features进行归一化
        _max = f.max(dim=1)[0]
        _min = f.min(dim=1)[0]

        norm = ((f.T - _min) / (_max - _min + 1e-5)).T
        return norm

    
    def collect_all_class(self, labels):
        # 收集所有样本的表征 
        features_all = self.features_all
        feature_dict = self._gen_features(detach=True)
        for layer in feature_dict.keys():
            features_l = feature_dict[layer]
            for idx, label in enumerate(labels):
                feature_l = features_l[idx]
                features_all[layer][label].append(feature_l)

    def gaussian_dis(self):
        # 计算均值和方差
        gaus_mean = {}
        gaus_std = {}
        for layer in self.features_all.keys():
            means = []
            stds = []
            for label in range(self.num_classes):
                mean = torch.stack(self.features_all[layer][label]).mean(dim=0)
                std = torch.stack(self.features_all[layer][label]).std(dim=0)
                means.append(mean)
                stds.append(std)
            gaus_mean[layer] = torch.stack(means)
            gaus_std[layer] = torch.stack(stds)
        
            # print('-' * 50)
            # print('mean.mean():', torch.stack(means).mean(dim=1))
            # print('std.mean():', torch.stack(stds).mean(dim=1))
            # print('-' * 50)

        self.gaus_mean, self.gaus_std = gaus_mean, gaus_std

        return gaus_mean, gaus_std
    
    def find_kernel(self, save_dir):
        # 找到鲁棒卷积核后保存数据
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.gaussian_dis()
        for layer in self.features_all.keys():
            means = self.gaus_mean[layer]
            stds = self.gaus_std[layer]
            stds = self._norm(stds)
            mask = torch.zeros(stds.shape)
            mask[torch.where(stds > self.alpha)] = 1

            print('-' * 50)
            print('Saving means and robust mask in layer{}'.format(layer))
            print('mask.shape', mask.shape)
            print('mask.sum', mask.sum())
            torch.save(means, os.path.join(save_dir, 'means_{}.pt'.format(layer)))
            torch.save(mask, os.path.join(save_dir, 'robust_{}.pt'.format(layer)))


    def remove_hook(self):
        for hookModule in self.hookModules:
            hookModule.remove_hook()


def norm(matrix, dim=1):
    mean = torch.mean(matrix, dim=dim)
    std = torch.std(matrix, dim=dim)
    n = matrix.sub_(mean[:, None]).div_(std[:, None])
    return n

