from collections import deque
from random import randint
from ddpg_networks import Actor, Critic
import numpy as np
import torch.optim as optim 
import torch


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, buffer_size, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = ReplayBuffer(buffer_size)

        self.critic = Critic(state_size, action_size, 0.01).to(device)
        self.critic_target = Critic(state_size, action_size, 0.01).to(device)
        self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=0.001, momentum=0.9)
        
        self.actor = Actor(state_size).to(device)
        self.actor_target = Actor(state_size).to(device)
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=0.001, momentum=0.9)
    
    def next_action(self, states):

        states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(states).cpu().data.numpy()
        self.actor.train()

        
        #if random.random() > epsilon:
        #    actions = np.argmax(candidates.cpu().data.numpy())
        #else:
        #actions = np.random.rand(1, self.action_size)
        
        actions = np.clip(action_values, -1, 1)  
        return actions

    def step(self, state, action, reward, next_state, done):
        ## Save in experience replay buffer
        self.memory.add( (state, action, reward, next_state, done) )

        ## If there is sufficient memories in the replay buffer
        if len(self.memory) > BATCH_SIZE:
            experience = self.memory.sample()

            self.learn(experience, GAMMA)

    def learn(self, experiences, gamma):
        
        state, action, reward, next_state, dones = experiences 

        ## Update Critic
        # Recovers next action from actor
        next_action = self.actor(next_state)

        # Train the critic using the experience (4)
        Q_target_next = self.critic_target(state, next_action)

        Q_target = reward + (GAMMA * Q_target_next * (1-dones))
    
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ## Update Actor
        new_action = self.actor(states)
        actor_loss = -self.critic_local(states, new_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

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

    def __init__(self, buffer_size):
        self.memory = deque(maxlen = buffer_size)
        self.buffer_size = buffer_size

    def add(self, e):
        self.memory.append(e)

    def sample(self):
        r = randint(0, self.buffer_size-1)
        return self.memory[r]




        
