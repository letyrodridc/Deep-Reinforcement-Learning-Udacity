import torch.nn as nn
import torch.nn.functional as F 
import torch
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module): 
    '''
    It's going to select the best action
    The action goes from -1 to 1 so, it uses tanh
    '''
    def __init__(self, state_size, action_size,seed,fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model."""
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_size, fcs1_units)
        self.linear1_bn = nn.BatchNorm1d(fcs1_units)
        self.linear2 = nn.Linear(fcs1_units, fc2_units)
        self.linear2_bn = nn.BatchNorm1d(fc2_units)
        self.linear3 = nn.Linear(fc2_units, action_size)
        
        self.reset_parameters()

    def forward(self,x):
        x1 = self.linear1_bn(F.relu(self.linear1(x)))
        x2 = self.linear2_bn(F.relu(self.linear2(x1)))
        x3 = self.linear3(x2)

        # Return the best-action
        return F.tanh(x3)
    
    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.linear1_bn = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.linear2_bn = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.linear1_bn(F.relu(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = self.linear2_bn(F.relu(self.fc2(x)))
        return self.fc3(x)

