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
