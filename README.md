# Echelon ML

A tensor and neural network library for threat intelligence, similar to PyTorch architecture.

## Installation

```bash
pip install -e .
```

## Features

- Tensor operations with numpy backend
- Automatic differentiation
- Neural network modules
- Optimizers (SGD, Adam)
- Dataset loading and processing

## Usage

```python
import echelon_ml as eml

# Create tensors
x = eml.tensor([1, 2, 3, 4])
y = eml.tensor([5, 6, 7, 8])

# Build neural networks
class Model(eml.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = eml.nn.Linear(10, 5)
        self.linear2 = eml.nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.linear1(x).relu()
        x = self.linear2(x).sigmoid()
        return x

# Train models
model = Model()
optimizer = eml.optim.Adam(model.parameters())
# ... training code ...
```

## License

MIT
