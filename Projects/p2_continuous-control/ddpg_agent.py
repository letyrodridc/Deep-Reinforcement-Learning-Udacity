import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_networks import Actor, Critic
from replay_buffer import ReplayBuffer
from ounoise import OUNoise
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, state_size, action_size, random_seed):
        """ Creates a new DDPG agent initilizing the networks """ 
        self.state_size = state_size
        self.action_size = action_size

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.critic = Critic(state_size, action_size, 17).to(device)
        self.critic_target = Critic(state_size, action_size, 17).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),  lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.actor = Actor(state_size, action_size, 17).to(device)
        self.actor_target = Actor(state_size, action_size, 17).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
    
        self.seed = random.seed(random_seed)
        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def next_action(self, states, add_noise=True):
        """ Returns the next action to take """
        states = torch.from_numpy(states).float().to(device)

        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(states).cpu().data.numpy()
        self.actor.train()

        actions = action_values
            
        if add_noise:
            actions += self.noise.sample()
        
        actions = np.clip(actions, -1, 1) 
            
        return actions

    def step(self, state, action, reward, next_state, done):
        """ Takes a next step saving the data in the replay buffer and learning new experiences """

        ## Save in experience replay buffer"""
        
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            self.memory.add(s, a, r, ns, d)


        ## If there is sufficient memories in the replay buffer
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()

            self._learn(experiences, GAMMA)

    def _learn(self, experiences, gamma):
        """ Learns new experiences """
        
        state, action, reward, next_state, dones = experiences 
        ## Update Critic
        # Recovers next action from actor
        next_action = self.actor(next_state)

        # Train the critic using the experience (4)
        Q_target_next = self.critic_target(next_state, next_action)

        Q_target = reward + (gamma * Q_target_next * (1 - dones))
    
        Q_expected = self.critic(state, action)

        critic_loss = F.mse_loss(Q_expected, Q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        ## Update Actor
        new_action = self.actor(state)
        actor_loss = -self.critic(state, new_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()


        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)





