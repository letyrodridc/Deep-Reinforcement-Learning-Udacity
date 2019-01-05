#### Leticia Lorena Rodríguez - Decembre 2018 - Deep Reinforcement Learning Nanodegree - Udacity

## Report: Continous Control Project

### Introduction

This project solves the Unity3D environment Rearcher for 20 agents. 

### Solution

It's solve the environment using a DDPG Agent. The agent uses Actor and Critic networks defined in the following way:

![Net](net.png)

The DDPG Algorithm is defined:

![DDPG](ddpg.png)

The parameters used are:

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 50        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
```

I've tried with different learning rate but the training just went slow without any increase. 

Also, I've used 2048 as input and output size in the network without great results.

I've noticed that learning rate has a strong impact in learning. 

The plot of the rewards is:

![Plot](plot.png)


It's showing that after around 20 episodes the average reward is more than 30 as expected.

The project also includes when the solution goes above 30 for the last 100 episodes:

Average score in the latest 100 episodes: 32.36344927662052

After 101 episodes. 

 ### Future work

Some ideas for future work are:
 * Try solving one agent with DDPG
 * Try implementing PPO, A3C, and D4PG
 * Tunning network training: changing learning rates, optimizers, etc.
