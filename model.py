import torch 
import torch.nn as nn

"""
Inputs of model are 
    - 49 firsts blocks of the snake 
    - apple position
"""

class Model(nn.Module):
    """
    Model class
    2 hidden layers of 100 neurons each
    last layer of 4 neurons for the 4 possible actions with softmax activation
    """
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(91, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

