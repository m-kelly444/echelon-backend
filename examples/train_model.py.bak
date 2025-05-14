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
