## Authors: Artur Filipowicz
## Version: 0.9
## Copyright: 2020
## MIT License

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from DQL import DQL_Agent

# Source: https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Environment information:
# https://github.com/openai/gym/wiki/MountainCar-v0
env = gym.make('MountainCar-v0')

"""
The observation is a 2-tuple of: 
position between -1.2 and 0.6 
velocity between -0.07 and 0.07

The actions are push left (0), no push (1), and push right (2).

The reward is -1 for each time step until the goal position of 0.5 is reached.

Car starts at random position from -0.6 to -0.4 with no velocity, and has 
200 iterations to reach the goal position.      
"""
agent = DQL_Agent(2, 3)
agent.epsDecay = 20000
agent.gamma = 0.99
agent.batch_size = 32
net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
agent.changeNet(net)
agent.optimizer = torch.optim.Adam(agent.parameters())
agent.updateRefNetFreq = 200

steps = 200000
losses = []
episode_reward = 0
episode_rewards = []

state = env.reset()
episode_reward = 0
for step in range(1, steps + 1):
    agent.lifetime += 1
    action = agent.act(state, True, env.action_space.n)
    next_state, reward, done, _ = env.step(int(action))
    agent.memorize(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        episode_rewards.append(episode_reward)
        episode_reward = 0
        state = env.reset()

    if len(agent.memory) > agent.batch_size:
        loss = agent.study()
        losses.append(loss.data.numpy())

    # Show training process
    if step % 100 == 0:
        plt.clf()
        plt.subplot(121)
        plt.plot(moving_average(losses,n=1500))
        plt.xlabel("Step")
        plt.ylabel("Average Loss")
        plt.subplot(122)
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.pause(0.1)

    # Run one episode to show agent progress
    if step % 10000 == 0:
        state = env.reset()
        for t in range(200):
            env.render()
            action = agent.act(state, False, env.action_space.n)
            state, reward, done, info = env.step(int(action))
        state = env.reset()

state = env.reset()
while(1):
    for t in range(200):
        env.render()
        action = agent.act(state, False, env.action_space.n)
        state, reward, done, info = env.step(int(action))
    state = env.reset()
env.close()


