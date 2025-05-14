import torch
import torch.nn as nn
import torch.nn.functional as F
from echelon.ml.module import Module

class ThreatAttention(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.scale = output_dim ** 0.5
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        
        return torch.matmul(attn, v)

class APTGroupClassifier(Module):
    def __init__(self, input_dim, hidden_dim, num_groups):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.attention = ThreatAttention(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, num_groups)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.attention(x)
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        return torch.sigmoid(self.output(x))

class AttackVectorClassifier(Module):
    def __init__(self, input_dim, hidden_dim, num_vectors):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, num_vectors)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        return torch.sigmoid(self.output(x))
