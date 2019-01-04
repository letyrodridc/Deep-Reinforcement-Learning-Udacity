import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """ Initiates a new Actor Network """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.fc3_bn = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.fc4_bn = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 512)
        self.fc5_bn = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 512)
        self.fc6_bn = nn.BatchNorm1d(512)
        self.fc7 = nn.Linear(512, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ Reset network weights """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc6.weight.data.uniform_(*hidden_init(self.fc6))
        self.fc7.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ Performs forward in the Network """
        x = self.fc1_bn(F.relu(self.fc1(state)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = self.fc3_bn(F.relu(self.fc3(x)))
        x = self.fc4_bn(F.relu(self.fc4(x)))
        x = self.fc5_bn(F.relu(self.fc5(x)))
        x = self.fc6_bn(F.relu(self.fc6(x)))
        return F.tanh(self.fc7(x))


class Critic(nn.Module):
    """ Creates a Critic Network""" 

    def __init__(self, state_size, action_size, seed):
        """ Initiates a new Critic Network""" 
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size*2, 512)
        self.fc2 = nn.Linear(512 + (action_size*2), 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """ Starts the weights with random values"""
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc6.weight.data.uniform_(*hidden_init(self.fc6))
        self.fc7.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Performs NN forward"""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.fc7(x)
