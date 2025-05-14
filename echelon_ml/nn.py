import numpy as np
import torch
from typing import Dict, List, Any, Optional

class Parameter:
    def __init__(self, data, name=None):
        self.value = data
        self.name = name
        if hasattr(data, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name
    
    def update(self, data):
        self.value = data
        if hasattr(data, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name
    
    def __repr__(self):
        return repr(self.value)

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params
    
    def add_parameter(self, name, value):
        param = Parameter(value, name)
        self._parameters[name] = param
        return param
    
    def __setattr__(self, key, value):
        if isinstance(value, Parameter):
            self._parameters[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value
        else:
            super().__setattr__(key, value)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def train(self):
        self.training = True
        for module in self._modules.values():
            if hasattr(module, 'train'):
                module.train()
        return self
    
    def eval(self):
        self.training = False
        for module in self._modules.values():
            if hasattr(module, 'eval'):
                module.eval()
        return self

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        bound = 1 / np.sqrt(in_features)
        weight_data = torch.Tensor(out_features, in_features).uniform_(-bound, bound)
        self.weight = Parameter(weight_data)
        
        if bias:
            bias_data = torch.Tensor(out_features).uniform_(-bound, bound)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None
    
    def forward(self, input):
        output = input @ self.weight.value.t()
        if self.bias is not None:
            output += self.bias.value
        return output

class ReLU(Module):
    def forward(self, x):
        return torch.relu(x)

class Sigmoid(Module):
    def forward(self, x):
        return torch.sigmoid(x)
