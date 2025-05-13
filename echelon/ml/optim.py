import torch
from typing import List

class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
        
        for i, param in enumerate(self.parameters):
            if hasattr(param, 'value') and param.value is not None:
                self.velocity[i] = torch.zeros_like(param.value)
    
    def zero_grad(self):
        for param in self.parameters:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()
            elif hasattr(param, 'value') and hasattr(param.value, 'grad') and param.value.grad is not None:
                param.value.grad.zero_()
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if hasattr(param, 'value') and param.value is not None and param.value.grad is not None:
                if i in self.velocity:
                    self.velocity[i] = self.momentum * self.velocity[i] + param.value.grad
                    param.value.data -= self.lr * self.velocity[i]
                else:
                    param.value.data -= self.lr * param.value.grad

class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        
        self.m = {}
        self.v = {}
        
        for i, param in enumerate(self.parameters):
            if hasattr(param, 'value') and param.value is not None:
                self.m[i] = torch.zeros_like(param.value)
                self.v[i] = torch.zeros_like(param.value)
    
    def zero_grad(self):
        for param in self.parameters:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()
            elif hasattr(param, 'value') and hasattr(param.value, 'grad') and param.value.grad is not None:
                param.value.grad.zero_()
    
    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if hasattr(param, 'value') and param.value is not None and param.value.grad is not None:
                if i in self.m and i in self.v:
                    self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.value.grad
                    self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * param.value.grad.pow(2)
                    
                    m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                    v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                    
                    param.value.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
