import torch
import numpy as np
from echelon_ml.tensor import tensor
from echelon_ml.nn import Module, Linear, Parameter
import echelon_ml.optim as optim

# Define a simple neural network
class SimpleNN(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(2, 10)
        self.layer2 = Linear(10, 1)
    
    def forward(self, x):
        # Apply first layer with ReLU
        x = self.layer1(x)
        x = torch.relu(x)
        # Apply second layer with sigmoid
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x

# Create simple XOR dataset
X = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=torch.float32)

y = torch.tensor([
    [0],
    [1],
    [1],
    [0]
], dtype=torch.float32)

# Initialize model and optimizer
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = torch.nn.BCELoss()

# Train the model
print("Training neural network on XOR problem...")
for epoch in range(2000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/2000], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    predictions = model(X)
    predicted = (predictions > 0.5).float()
    accuracy = (predicted == y).float().mean().item()
    print(f"Final predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i].tolist()}, Target: {y[i].item()}, Prediction: {predictions[i].item():.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
