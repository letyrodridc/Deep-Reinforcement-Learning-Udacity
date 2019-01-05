from collections import namedtuple, deque
import numpy as np
import torch
import random 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:

    """
    Creates a replay buffer to store all the experiences
    """
    
    def __init__(self,  buffer_size, batch_size, seed, agents_qty):
        """ Creates a replay buffer of buffer_size """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.agents_qty = agents_qty
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Get random experiences from the buffer of size batch_size """
        experiences = random.sample(self.memory, k=self.batch_size)

        states_list = list()
        actions_list = list()
        rewards_list = list()
        next_states_list = list()
        dones_list = list()

        
        for i in range(self.agents_qty):
            idx = np.array([i])
            states = torch.from_numpy(np.vstack([e.states[idx] for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.actions[idx] for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_states[idx] for e in experiences if e is not None])).float().to(device)
   
            states_list.append(states)
            actions_list.append(actions)
            next_states_list.append(next_states)
   
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states_list, actions_list, rewards, next_states_list, dones)

    def __len__(self):
        """ Returns the quantity of elements stored in the buffer """
        return len(self.memory)