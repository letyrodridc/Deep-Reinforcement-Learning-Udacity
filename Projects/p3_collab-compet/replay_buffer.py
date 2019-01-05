from collections import namedtuple, deque
import numpy as np
import torch
import random 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    
    def __init__(self,  buffer_size, batch_size, seed):
        """ Creates a replay buffer of buffer_size """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done",
                                                  "others_states",
                                                  "others_actions",
                                                  "others_next_states"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done, others_states,
            others_actions, others_next_states):
        e = self.experience(state, action, reward, next_state, done,
                            others_states, others_actions, others_next_states)
        self.memory.append(e)

    def sample(self):
        """ Get random experiences from the buffer of size batch_size """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        others_states = torch.from_numpy(np.vstack([e.others_states for e in experiences
                                  if e is not None])).float().to(device)
        others_actions = torch.from_numpy(np.vstack([e.others_actions for e in experiences
                                   if e is not None])).float().to(device)
        others_next_states = torch.from_numpy(np.vstack([e.others_next_states
                                                      for e in experiences
                                                      if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones, others_states,
                others_actions, others_next_states)

    def __len__(self):
        """ Returns the quantity of elements stored in the buffer """
        return len(self.memory)