#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from models import Actor, Critic
from replay_buffer import ReplayBuffer
from ounoise import OUNoise

BUFFER_SIZE = 100000  # replay buffer size
BATCH_SIZE = 50        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR_ACTOR = 0.0001        # learning rate of the actor 
LR_CRITIC = 0.001        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    '''
    MultiAgent DDPG implementation
    '''

    def __init__(self, agents_qty, state_size, action_size, rand_seed):
        '''
        Creates a new instance of MultiAgent DDPG agent
        
        agent_qty: indicates the number of agents.
        state_size: size of the observation space
        action_size: size of the actions space
        '''    

        self.agents_qty = agents_qty
        
        # Multiple Replay Buffer creation
        self.memory = ReplayBuffer(BUFFER_SIZE,
                                   BATCH_SIZE, rand_seed, agents_qty)
     
        self.action_size = action_size
        self.state_size = state_size 
        self.agents = [DDPGAgent(i, self.state_size,
                               self.action_size,
                               rand_seed,
                               self)
                         for i in range(self.agents_qty)]

    def step(self, states, actions, rewards, next_states, dones):
        """ Add into memory the experience. It will include all agents information """

        self.memory.add(states, actions, rewards, next_states, dones)

        # For each agent perfoms a step 
        for agent in self.agents:
            agent.step()

    def act(self, states, add_noise=True):
        """ Returns the action to take """
        na_rtn = np.zeros([self.agents_qty, self.action_size])
        for agent in self.agents:
            idx = agent.agent_id
            na_rtn[idx, :] = agent.act(states[idx], add_noise)
        return na_rtn

    def save_weights(self):
        """ Save agents weights"""
        for agent in self.agents:
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(agent.agent_id+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(agent.agent_id+1))
    
    def reset(self):
        """ Just reset all agents """
        for agent in self.agents:
            agent.reset()

    def __len__(self):
        return self.agents_qty

    def __getitem__(self, key):
        return self.agents[key]


class DDPGAgent:
    '''
    DDPG Agent implementation
    '''

    def __init__(self, agent_id, state_size, action_size, rand_seed, meta_agent):
        """ Creates a new DDPG Agent """

        self.agent_id = agent_id
        self.action_size = action_size

        # Defines the Actor Networks 
        self.actor_local = Actor(state_size, action_size, rand_seed).to(device)
        self.actor_target = Actor(state_size, action_size, rand_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Defines the Critic Networks 
        self.critic_local = Critic(state_size,
                                   action_size,
                                   meta_agent.agents_qty,
                                   rand_seed).to(device)
        self.critic_target = Critic(state_size,
                                    action_size,
                                    meta_agent.agents_qty,
                                    rand_seed).to(device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC )#, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size, rand_seed)

        # Refers to the MA agent memory
        self.memory = meta_agent.memory

        self.t_step = 0

    def step(self):
        
        # Takes an step
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        (states_list, actions_list, rewards, next_states_list, dones) = experiences

        # Get the target actions for all the states
        l_all_next_actions = []
        for states in states_list:
            l_all_next_actions.append(self.actor_target(states))
        
        # Convert the experiences into Torch tensors
        all_next_actions = torch.cat(l_all_next_actions, dim=1).to(device)
        all_next_states = torch.cat(next_states_list, dim=1).to(device)
        all_states = torch.cat(states_list, dim=1).to(device)
        all_actions = torch.cat(actions_list, dim=1).to(device)

        
        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        
        # Calculates the Q function using all the next states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --------------------------- update actor ---------------------------
        actions_pred = []
        for states in states_list:
            actions_pred.append(self.actor_local(states))
        
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        

        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- update target networks ----------------------
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def act(self, states, add_noise=True):
        """ Returns the actions to take by the agent"""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()


    def soft_update(self, local_model, target_model, tau):
        """ Performs the softupdate """
        iter_params = zip(target_model.parameters(), local_model.parameters())
        for target_param, local_param in iter_params:
            tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data
            target_param.data.copy_(tensor_aux)

    def reset(self):
        self.noise.reset()

