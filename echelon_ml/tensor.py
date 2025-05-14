import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Union, Optional

class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, torch.Tensor):
            self.data = data.detach().numpy()
        else:
            self.data = np.array(data, dtype=np.float32)
        
        self.grad = None if not requires_grad else np.zeros_like(self.data)
        self._requires_grad = requires_grad
        self._history = None
        self.shape = self.data.shape
        self.size = self.data.size
    
    def requires_grad_(self, requires_grad=True):
        self._requires_grad = requires_grad
        if requires_grad and self.grad is None:
            self.grad = np.zeros_like(self.data)
        return self
    
    def backward(self, grad=None):
        if grad is None:
            if self.size == 1:
                grad = np.ones_like(self.data)
            else:
                raise ValueError("Must specify gradient for non-scalar tensors")
        
        if self._history is not None:
            pass
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)
    
    def __pow__(self, power):
        return Tensor(self.data ** power)
    
    def __neg__(self):
        return Tensor(-self.data)
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return Tensor(self.data - other)
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)
    
    def sum(self, dim=None):
        return Tensor(self.data.sum(axis=dim))
    
    def mean(self, dim=None):
        return Tensor(self.data.mean(axis=dim))
    
    def log(self):
        return Tensor(np.log(self.data))
    
    def exp(self):
        return Tensor(np.exp(self.data))
    
    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.data)))
    
    def relu(self):
        return Tensor(np.maximum(0, self.data))
    
    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape))
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self._requires_grad})"

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)
