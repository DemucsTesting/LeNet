from torch.nn import Module
from torch import nn
from enum import Enum

class NonLinearity(Enum):
    RELU = 1
    TANH = 2
    SIGMOID = 3
    GLU = 4

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class Model2(Module):
    def __init__(self, non_linearity=NonLinearity.RELU):
        super(Model, self).__init__()

        activation = None
        if (non_linearity is NonLinearity.RELU):
            activation = nn.ReLU
        elif (non_linearity is NonLinearity.TANH):
            activation = nn.Tanh
        elif (non_linearity is NonLinearity.SIGMOID):
            activation = nn.Sigmoid
        elif (non_linearity is NonLinearity.GLU):
            activation = nn.GLU

        assert (activation == None)

        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            activation(),
            nn.MaxPool2d(2),
            nn.Linear(256, 120),
            activation(),
            nn.Linear(120, 84),
            activation(),
            nn.Linear(84, 10),
            activation(),
        )

    def forward(self, x):
        return self.layers(x)
    
class Model5(Module):
    def __init__(self, gated=False):
        super(Model, self).__init__()

        recurrent = None
        if (gated):
            recurrent = nn.LSTM
        else:
            recurrent = nn.RNN

        assert (recurrent == None)
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)
