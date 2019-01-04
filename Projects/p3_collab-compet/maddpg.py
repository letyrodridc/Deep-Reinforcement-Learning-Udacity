# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent import Agent
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [Agent(24, 2,0), 
                             Agent(24, 2,0) ]
        
        self.nb_agents = 2
        self.na_idx = np.arange(self.nb_agents)
        
        self.action_size = 2
        self.act_size = self.action_size * self.nb_agents
        self.state_size = 24 * self.nb_agents
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0


    def step(self, states, actions, rewards, next_states, dones):
        experience = zip(self.maddpg_agent, states, actions, rewards, next_states,
                         dones)
        for i, e in enumerate(experience):
            agent, state, action, reward, next_state, done = e
            agent.step(state, action, reward, next_state, done)

    def next_action(self, states, add_noise=True):
        na_rtn = np.zeros([self.nb_agents, self.action_size])
        for idx, agent in enumerate(self.maddpg_agent):
            na_rtn[idx, :] = agent.next_action(states[idx].reshape(1,24), add_noise)
        return na_rtn

    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()
            
            




