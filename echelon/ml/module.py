import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union

class Parameter:
    def __init__(self, data, name=None):
        self.value = data
        self.name = name
        self.metadata = {}
    
    def zero_grad(self):
        if hasattr(self.value, 'grad') and self.value.grad is not None:
            self.value.grad.zero_()

class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self._echelon_parameters = {}
        self.training = True
    
    def add_parameter(self, name: str, value: torch.Tensor) -> Parameter:
        param = Parameter(nn.Parameter(value), name)
        self._echelon_parameters[name] = param
        setattr(self, name, param.value)
        return param
    
    def parameters(self) -> List[Parameter]:
        params = list(self._echelon_parameters.values())
        for name, module in self.named_children():
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params

    def train(self):
        self.training = True
        for module in self.children():
            if hasattr(module, 'train'):
                module.train()
                
    def eval(self):
        self.training = False
        for module in self.children():
            if hasattr(module, 'eval'):
                module.eval()
