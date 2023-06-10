import sys
sys.path.append('/home/gjs/code/robust_eval')
import torch
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
import os
from util import load_modules

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

 
class HookModule_partial:
    def __init__(self, module, layer, value_type='+'):
        self.module = module
        self.value_type = value_type
        self.matrix = None
        self.layer = layer
        self.hook = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.outputs = outputs
        if isinstance(module, nn.Conv2d):
            # print("inputs[0]:", inputs[0].shape)
            partial_outputs = self.partial_conv(inputs[0], module, outputs.size(2), outputs.size(3))  # [b, o, i, h, w]
            if self.value_type == '+':
                partial_outputs = torch.relu(torch.sum(partial_outputs, dim=(3, 4)))  # [b, o, i]
        if isinstance(module, nn.Linear):
            partial_outputs = self.partial_linear(inputs[0], module)  # [b, o, i]
            if self.value_type == '+':
                partial_outputs = torch.relu(partial_outputs)  # [b, o, i]
        self.p_outputs = partial_outputs
        self.matrix = partial_outputs.sum(dim=2)
        
    
    # 卷积操作的中间值
    def partial_conv(self, inp: torch.Tensor, conv: nn.Conv2d, o_h=None, o_w=None):
        kernel_size = conv.kernel_size
        dilation = conv.dilation
        padding = conv.padding
        stride = conv.stride
        weight = conv.weight  # O I K K
        bias = conv.bias  # O

        # print("weight.shape: ", weight.shape)

        wei_res = weight.view(weight.size(0), weight.size(1), -1).permute((1, 2, 0))  # I K*K O
        inp_unf = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)(inp)  # B K*K N
        inp_unf = inp_unf.view(inp.size(0), inp.size(1), wei_res.size(1), o_h, o_w)  # B I K*K H_O W_O
        out = torch.einsum('ijkmn,jkl->iljmn', inp_unf, wei_res)  # B O I H W
        # out = out.sum(2)
        # bias = bias.unsqueeze(1).unsqueeze(2).expand((out.size(1), out.size(2), out.size(3)))  # O H W
        # out = out + bias
        return out


    def partial_linear(self, inp: torch.Tensor, linear: nn.Linear):
        # inp: B I
        weight = linear.weight  # [O I]
        bias = linear.bias  # O

        B = inp.shape[0]
        I = weight.shape[1]
        O = weight.shape[0]

        inp = inp.unsqueeze(1).expand(B, O, I)
        weight = weight.unsqueeze(0).expand(B, O, I)
        out = inp * weight
        # out = out.sum(2)
        # out = out + bias
        return out  # [B O I]

    def remove_hook(self):
        self.hook.remove()


class HookModule:
    '''
    表征热力图可视化
    1. 在执行model(x)之前注册HookModule(for循环外)
    2. 在执行model(x)之后调用collect_函数收集表征信息(for循环内)
    3. 在模型运算结束后调用view函数保存热力图信息(for循环外)
    4. 在模型运算结束后调用函数remove_hook(for循环外)
    '''
    def __init__(self, model, layers, partial=False):
        modules = load_modules(model, layers)
        self.hookModules = []
        self.features_t = {} # 同类样本表征
        for idx, module in enumerate(modules):
            if partial == False:
                # 正常的中间层表征输出
                hookModule = HookModule_forward(module=module, layer=layers[idx])
            else:
                # 经过处理的中间层表征输出
                hookModule = HookModule_partial(module=module, layer=layers[idx])
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

    def collect_same_class(self, labels, class_t=0):
        # 收集同类样本的表征
        features_t = self.features_t
        feature_dict = self._gen_features(detach=True)
        for layer in feature_dict.keys():
            features_l = feature_dict[layer]
            for label in labels:
                if label.item() != class_t:
                    continue
                feature_l = features_l[label]
                if layer in features_t.keys():
                    features_t[layer] = torch.cat((features_t[layer], feature_l.unsqueeze(0)), dim=0)
                else:
                    features_t[layer] = feature_l.unsqueeze(0)
        return features_t
    
    def view_features(self, features, save_name):
        # 将features处理后调用_gen_heatmap进行可视化
        for layer in features.keys():
            random_idx = np.random.choice(features[layer].shape[0], 20, replace=False).tolist()
            feature = features[layer][random_idx] # 随机筛选一部分样本进行可视化
            # feature = norm(feature, dim=1)
            save_path = os.path.join(config.images_dir, save_name + '_{}.png'.format(layer))
            self._gen_heatmap(feature.detach().cpu(), save_path)

    def _gen_heatmap(self, feature, save_path):
        # 对shape为b_*c的feature以热力图的格式进行可视化
        f, ax = plt.subplots(figsize=(512, 10), ncols=1)
        ax.set_xlabel('convolutional kernel')
        ax.set_ylabel('category')
        sns.heatmap(feature, annot=False, ax=ax)
        plt.savefig(save_path, bbox_inches='tight')
        # plt.show()
        plt.clf()

    def remove_hook(self):
        for hookModule in self.hookModules:
            hookModule.remove_hook()


def norm(matrix, dim=1):
    mean = torch.mean(matrix, dim=dim)
    std = torch.std(matrix, dim=dim)
    n = matrix.sub_(mean[:, None]).div_(std[:, None])
    return n

