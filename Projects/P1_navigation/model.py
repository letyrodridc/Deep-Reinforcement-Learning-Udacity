import torch
import torch.nn as nn
import torch.nn.functional as F

class My_DeepQ_Network(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(My_DeepQ_Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)
        "*** YOUR CODE HERE ***"

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state.view(-1, self.state_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
     
        

