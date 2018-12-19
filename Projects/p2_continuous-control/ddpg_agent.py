from collections import deque
from random import randint
from ddpg_networks import Actor, Critic
import numpy as np
import torch.optim as optim 
import torch
import torch.nn.functional as F
import random
import copy


#BUFFER_SIZE = 10        #int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, buffer_size, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = ReplayBuffer(buffer_size, BATCH_SIZE)

        self.critic = Critic(state_size, action_size, 2).to(device)
        self.critic_target = Critic(state_size, action_size, 2).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        self.actor = Actor(state_size, action_size, 2).to(device)
        self.actor_target = Actor(state_size, action_size, 2).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
    
        self.seed = random.seed(random_seed)
        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def next_action(self, states, add_noise=True):
        states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(states).cpu().data.numpy()
        self.actor.train()

        #if add_noise:
        #    action_values += self.noise.sample()
        
        actions = np.clip(action_values, -1, 1) 
        return actions[0]

    def step(self, state, action, reward, next_state, done):
        ## Save in experience replay buffer
        self.memory.add( (state, action, reward, next_state, done) )

        ## If there is sufficient memories in the replay buffer
        if self.memory.length() > BATCH_SIZE:
            experiences = self.memory.sample()

            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        
        state, action, reward, next_state, dones = experiences 
        ## Update Critic
        # Recovers next action from actor
        next_action = self.actor(next_state)

        # Train the critic using the experience (4)
        Q_target_next = self.critic_target(next_state, next_action)

        Q_target = reward + (gamma * Q_target_next * (1-dones))
    
        Q_expected = self.critic(state, action)

        critic_loss = F.mse_loss(Q_expected, Q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ## Update Actor
        new_action = self.actor(state)
        actor_loss = -self.critic(state, new_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen = buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def add(self, e):
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def length(self):
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
        
