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
env = gym.make('CartPole-v0')

"""
The observation is a 4-tuple of: 
position between -2.4 and 2.4 
velocity between -inf and inf
pole angle between -41.8 and 41.8
pole velocity at tip between -inf and inf

The actions are push left (0), and push right (1).

The reward is 1 for each time step until:
1) pole angle is more than ±12°
2) cart position is more than ±2.4
3) episode length is greater than 200

Cart starts with all observations assigned a uniform 
random value between ±0.05.      
"""
agent = DQL_Agent(4, 2)
agent.epsDecay = 1500
agent.gamma = 0.999
agent.batch_size = 64
net = nn.Sequential(
            nn.Linear(4, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 2),
        )
agent.changeNet(net)
agent.optimizer = torch.optim.Adam(agent.parameters())
agent.updateRefNetFreq = 1

steps = 5000
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
    if step % 50 == 0:
        plt.clf()
        plt.subplot(121)
        plt.plot(moving_average(losses,n=20))
        plt.xlabel("Step")
        plt.ylabel("Average Loss")
        plt.subplot(122)
        plt.plot(moving_average(episode_rewards, n=20))
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.pause(0.1)

    # Run one episode to show agent progress
    if step % 500 == 0:
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


