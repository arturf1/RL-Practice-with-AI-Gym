## Authors: Artur Filipowicz
## Version: 0.9
## Copyright: 2020
## MIT License

import gym
import matplotlib.pyplot as plt
import numpy as np
from DQL import DQL_Agent
import torch
import torch.nn as nn
from torch.autograd import Variable

# Source: https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Environment information:
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/roulette.py
env = gym.make('Roulette-v0')

"""
The observation is a 1-tuple of: always zero.

The actions are bet (0-36) or stop (37).

The reward for betting 0 and 0 comes up is 35. Reward for matching parity is +1, 
and losing is -1.    
"""
agent = DQL_Agent(1, 38)
agent.epsDecay = 300
agent.gamma = 0.0 # each step is independent
agent.batch_size = 64
net = nn.Sequential(
            nn.Linear(1, 38)
        )
agent.changeNet(net)
agent.optimizer = torch.optim.Adam(agent.parameters())
agent.updateRefNetFreq = 1

steps = 10000
losses = []
episode_reward = 0
episode_rewards = []

state = env.reset()
episode_reward = 0
for step in range(1, steps + 1):
    agent.lifetime += 1
    action = agent.act(state, True, env.action_space.n)
    next_state, reward, done, _ = env.step(int(action))
    print([state], action, reward, [next_state], done)
    agent.memorize([state], action, reward, [next_state], done)

    state = next_state
    episode_reward += reward

    if done:
        episode_rewards.append(episode_reward)
        episode_reward = 0
        state = float(env.reset())

    if len(agent.memory) > agent.batch_size:
        loss = agent.study()
        losses.append(loss.data.numpy())

    # Show training process
    if step % 50 == 0:
        plt.clf()
        plt.subplot(121)
        plt.plot(moving_average(losses,n=300))
        plt.xlabel("Step")
        plt.ylabel("Average Loss")
        plt.subplot(122)
        plt.plot(moving_average(episode_rewards, n=10))
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.pause(0.1)

# Check action of network for each state
plt.figure()
state = Variable(torch.FloatTensor([0]))
Q = agent.net(state)
print(Q.detach().numpy())
plt.bar(np.arange(0,38),Q.detach().numpy())
plt.xlabel("Action")
plt.ylabel("Q")
plt.show()
