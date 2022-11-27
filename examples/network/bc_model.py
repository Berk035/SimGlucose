import torch
import torch.nn as nn
import torch.nn.functional as F

class BC(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(BC,self).__init__()
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_a = nn.Linear(hidden_dim, action_dim)

    def forward(self,x):
        x = self.linear_1(x)
        x = F.leaky_relu(x, 0.001)
        x = self.linear_2(x)
        x = F.leaky_relu(x, 0.001)
        x = self.linear_3(x)
        x = F.leaky_relu(x, 0.001)
        x_a = self.linear_a(x)

        return x_a
        
