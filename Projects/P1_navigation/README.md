## Deep Reinforcement Learning Beta Testing - Project 1 
## Letyrodri - Collect banans agent implementation

## Project Details

The environment consist in a game that the agent need to collect the yellow bananas and not the blue ones.

Each time a yellow banana is properly collected the agent is going to receive a +1 reward. Meanwhile, a -1 reward will be received is the banana collected is blue.

To pass the game, the agent should score +13 in a consecutive sequence of 100 episodes. That's when the environment is going to be considered solved.

The _state space_ has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 

The _action space_ consist an action with this possible value:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn righ

Using this information, the agent is going to decide the best next action.

## Getting Started
 	
The project was coded using Pytorch and Jupyter notebooks. 

It's a necessary environment configuration that you will need. A requirements.txt is provided to easily configure it. Just install Anaconda and write:

```
conda create -r requirements.txt
``` 

To install the simulator, you need to download it from: 
*[Linux simulator](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
*[MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
*[Windows 32bits](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
*[Windows 64bits](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Instructions

A checkpoint.pth file is provided. It's necessary to run the agent and it contains the weights calculated during training that allows the agent to calculate the best action to take next. The checkpoint.pth is loaded over a Pytorch model.

The model was trained using Pytorch. For retraining the model you will need to have a Pytorch environment configured. It's not necessary to use GPU. The CPU training takes around 30 minutes for the provided configuration (2000 episodes).

To train the model and read the details of the implementation, use Jupyter notebooks to open the report.ipynb file. It constains the code that simulates the states and perform the training, and further explanation of the algoritm DQN used.

To see the model running, launch the simulator and execute the code in the jupyter notebook. Notice that "training" variable should be in "False" to not retrain the model and use the saved checkpoint.pth file.
