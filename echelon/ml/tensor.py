import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional, Union

class EchelonTensor:
    def __init__(self, data, requires_grad=False, metadata=None):
        if isinstance(data, torch.Tensor):
            self.data = data
        else:
            self.data = torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad)
        
        self.metadata = metadata or {}
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        
    def __repr__(self):
        return f"EchelonTensor(shape={self.shape}, grad={self.requires_grad}, metadata={list(self.metadata.keys())})"
    
    def add_metadata(self, key, value):
        self.metadata[key] = value
        return self
    
    def backward(self):
        if self.requires_grad:
            self.data.backward()
    
    def __add__(self, other):
        if isinstance(other, EchelonTensor):
            new_metadata = {**self.metadata}
            for k, v in other.metadata.items():
                if k not in new_metadata:
                    new_metadata[k] = v
                elif isinstance(new_metadata[k], list) and isinstance(v, list):
                    new_metadata[k] = list(set(new_metadata[k] + v))
            
            result = EchelonTensor(
                self.data + other.data,
                requires_grad=self.requires_grad or other.requires_grad,
                metadata=new_metadata
            )
            return result
        else:
            return EchelonTensor(
                self.data + other,
                requires_grad=self.requires_grad,
                metadata=self.metadata
            )
    
    @classmethod
    def from_text(cls, text, vectorizer):
        vec = vectorizer.transform([text]).toarray()
        return cls(torch.tensor(vec, dtype=torch.float32))
