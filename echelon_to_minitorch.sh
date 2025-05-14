#!/bin/bash

# Convert Echelon to MiniTorch-like structure
set -e

mkdir -p echelon_ml/{tensor,autodiff,optim,nn,data}
mkdir -p examples
mkdir -p tests

# Create root structure
echo 'from echelon_ml.tensor import Tensor
from echelon_ml.autodiff import Variable, Context, central_difference
from echelon_ml.optim import SGD, Adam
from echelon_ml.nn import Module, Parameter, Linear, ReLU, Sigmoid
from echelon_ml.data import ThreatDataset, load_data' > echelon_ml/__init__.py

# Core tensor implementation
cat > echelon_ml/tensor.py << 'EOL'
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
EOL

# Autodiff implementation
cat > echelon_ml/autodiff.py << 'EOL'
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class Variable:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name or ""
        self.derivative = None
        self.is_constant = False
        self.history = None
    
    def accumulate_derivative(self, derivative):
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += derivative
    
    def is_leaf(self):
        return self.history is None or self.history.last_fn is None
    
    def backward(self, derivative=None):
        if derivative is None:
            derivative = 1.0
        backpropagate(self, derivative)

class Context:
    def __init__(self, no_grad=False):
        self.no_grad = no_grad
        self.saved_values = ()
    
    def save_for_backward(self, *values):
        if self.no_grad:
            return
        self.saved_values = values
    
    @property
    def saved_tensors(self):
        return self.saved_values

def backpropagate(variable, deriv):
    if variable.is_leaf():
        variable.accumulate_derivative(deriv)
        return
    
    if variable.history is None:
        return
    
    for next_var, grad in variable.history.chain_rule(deriv):
        backpropagate(next_var, grad)

def central_difference(f, *vals, arg=0, epsilon=1e-6):
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    return (f(*vals1) - f(*vals2)) / (2.0 * epsilon)
EOL

# NN implementation
cat > echelon_ml/nn.py << 'EOL'
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
EOL

# Optim implementation
cat > echelon_ml/optim.py << 'EOL'
import torch
from typing import List, Dict, Any, Iterable

class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0):
        self.parameters = list(parameters)
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
        self.parameters = list(parameters)
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
EOL

# Data module implementation - CORRECT version directly in data.py, not in a subfolder
cat > echelon_ml/data.py << 'EOL'
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.model_selection import train_test_split

class ThreatDataset:
    def __init__(self, data_source, test_size=0.2, random_state=42):
        self.data_source = data_source
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.features = None
        self.labels = None
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        
        self._load_data()
        self._split_data()
    
    def _load_data(self):
        if isinstance(self.data_source, str):
            if self.data_source.endswith('.csv'):
                self.data = pd.read_csv(self.data_source)
            elif self.data_source.endswith('.json'):
                self.data = pd.read_json(self.data_source)
            else:
                raise ValueError(f"Unsupported file format: {self.data_source}")
        elif isinstance(self.data_source, pd.DataFrame):
            self.data = self.data_source
        else:
            raise ValueError(f"Unsupported data source type: {type(self.data_source)}")
        
        self._preprocess_data()
    
    def _preprocess_data(self):
        if 'label' in self.data.columns:
            self.labels = self.data['label'].values
            self.features = self.data.drop('label', axis=1).values
        else:
            raise ValueError("Data must contain a 'label' column")
    
    def _split_data(self):
        if self.features is None or self.labels is None:
            raise ValueError("Data must be loaded and preprocessed before splitting")
        
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.features, self.labels, test_size=self.test_size, random_state=self.random_state
        )
    
    def __len__(self):
        return len(self.data) if self.data is not None else 0

def load_data(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.json'):
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
EOL

# Create setup.py
cat > setup.py << 'EOL'
from setuptools import setup, find_packages

setup(
    name="echelon_ml",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "joblib>=1.0.0",
    ],
    python_requires='>=3.8',
)
EOL

# Create sample examples
cat > examples/simple_tensor.py << 'EOL'
import torch
import echelon_ml as eml
import numpy as np

x = eml.tensor([1, 2, 3, 4])
y = eml.tensor([5, 6, 7, 8])

z = x + y
print(z)

a = eml.tensor([[1, 2], [3, 4]], requires_grad=True)
b = a * a + 3
print(b)
EOL

cat > examples/train_model.py << 'EOL'
import torch
import echelon_ml as eml
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, random_state=42)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y

input_dim = 20
hidden_dim = 128
output_dim = 1

class ThreatModel(eml.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = eml.nn.Linear(input_dim, hidden_dim)
        self.fc2 = eml.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = eml.nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = ThreatModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

epochs = 100
batch_size = 32

for epoch in range(epochs):
    indices = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_tensor[batch_indices]
        y_batch = y_tensor[batch_indices]
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_outputs = model(X_tensor)
    test_loss = criterion(test_outputs, y_tensor).item()
    predictions = (test_outputs >= 0.5).float()
    accuracy = (predictions == y_tensor).float().mean().item()
    
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), 'models/threat_model.pt')
EOL

# Create test module
mkdir -p tests
cat > tests/test_tensor.py << 'EOL'
import unittest
import numpy as np
import torch
import echelon_ml as eml

class TestTensor(unittest.TestCase):
    def test_creation(self):
        t1 = eml.tensor([1, 2, 3, 4])
        self.assertEqual(t1.shape, (4,))
        self.assertEqual(t1.size, 4)
        
        t2 = eml.tensor(np.array([[1, 2], [3, 4]]))
        self.assertEqual(t2.shape, (2, 2))
        self.assertEqual(t2.size, 4)
    
    def test_operations(self):
        a = eml.tensor([1, 2, 3])
        b = eml.tensor([4, 5, 6])
        
        c = a + b
        np.testing.assert_array_equal(c.data, np.array([5, 7, 9]))
        
        d = a * b
        np.testing.assert_array_equal(d.data, np.array([4, 10, 18]))
        
        e = a - b
        np.testing.assert_array_equal(e.data, np.array([-3, -3, -3]))
        
        f = a / b
        np.testing.assert_array_almost_equal(f.data, np.array([0.25, 0.4, 0.5]))

if __name__ == '__main__':
    unittest.main()
EOL

# Directory structure
mkdir -p models data

echo "✅ Conversion complete. The repository now has a structure similar to MiniTorch."
echo "📁 Directory structure:"
find . -type d -not -path "*/\.*" | sort

echo "🔧 Run 'pip install -e .' to install the package."
