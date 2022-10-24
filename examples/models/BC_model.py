import torch.nn as nn
import torch
import torch.nn.functional as F

class NeuralNet(nn.Module):

    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.fc3 = nn.Linear(hidden_size2,output_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x.view(x.size(0),-1)