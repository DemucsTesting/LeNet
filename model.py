import torch

from torch.nn import Module
from torch import nn
from enum import Enum

class NonLinearity(Enum):
    RELU = 0
    TANH = 1
    SIGMOID = 2
    # GLU = 3 # (Reduces size. can't use directly).

class RecType(Enum):
    LINEAR = 0
    RNN = 1
    LSTM = 2

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.in_layer(x)
        y = y.view(y.shape[0], -1)
        y = self.out_layer(y)
        return y

class InequalityModel(Module):
    def __init__(self, non_linearity=NonLinearity.RELU):
        super(InequalityModel, self).__init__()

        activation = None
        if (non_linearity is NonLinearity.RELU):
            activation = nn.ReLU
        elif (non_linearity is NonLinearity.TANH):
            activation = nn.Tanh
        elif (non_linearity is NonLinearity.SIGMOID):
            activation = nn.Sigmoid
        # elif (non_linearity is NonLinearity.GLU):
        #     activation = nn.GLU

        assert (activation != None)

        self.in_layer = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            activation(),
            nn.MaxPool2d(2),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(256, 120),
            activation(),
            nn.Linear(120, 84),
            activation(),
            nn.Linear(84, 10),
            activation(),
        )

    def forward(self, x):
        y = self.in_layer(x)
        y = y.view(y.shape[0], -1)
        y = self.out_layer(y)
        return y
    
class RecurrentModel(Module):
    def __init__(self, type=RecType.LINEAR):
        super(RecurrentModel, self).__init__()

        recurrence = None
        self.h_0 = None
        self.recurrent = False

        if (type is RecType.LINEAR):
            recurrence = nn.Linear
            self.recurrent = False
        elif (type is RecType.RNN):
            self.h_0 = torch.zeros((1,84))
            recurrence = nn.RNN
            self.recurrent = True
        elif (type is RecType.LSTM):
            self.h_0 = (torch.zeros((1,84)), torch.zeros((1,84)))
            recurrence = nn.LSTM
            self.recurrent = True

        assert (recurrence != None)
        
        self.in_layer = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pre_layer = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
        )

        self.r_layer = recurrence(120,84)

        self.pos_layer = nn.Sequential(
            nn.Linear(84, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.in_layer(x)
        y = y.view(y.shape[0], -1)

        y = self.pre_layer(y)

        if (self.recurrent):
            y, _ = self.r_layer(y, self.h_0)
        else:
            y = self.r_layer(y)

        y = self.pos_layer(y)
        return y
