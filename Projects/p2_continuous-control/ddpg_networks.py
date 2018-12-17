import torch.nn as nn
import torch.nn.functional as F 
import torch
class Actor(nn.Module): 
	'''
	It's going to select the best action
	The action goes from -1 to 1 so, it uses tanh
	'''
	def __init__(self, state_size):
		"""Initialize parameters and build model."""
		super(Actor, self).__init__()

		self.linear1 = nn.Linear(state_size, 64)
		self.linear2 = nn.Linear(64, 4)

	def forward(self,x):

		x1 = F.relu(self.linear1(x))
		x2 = F.relu(self.linear2(x1))

		return F.tanh(x2)

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
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

