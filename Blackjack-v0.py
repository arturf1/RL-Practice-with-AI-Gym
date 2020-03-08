## Authors: Artur Filipowicz
## Version: 0.9
## Copyright: 2020
## MIT License

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import sleep
from DQL import DQL_Agent

# Source: https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Environment information:
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
env = gym.make('Blackjack-v0')

"""
The observation is a 3-tuple of: the player's current sum,
the dealer's one showing card (1-10 where 1 is ace),
and whether or not the player holds a usable ace (0 or 1).

The actions are request additional cards (hit=1) or stop (stick=0).

The reward for winning is +1, drawing is 0, and losing is -1.    
"""
agent = DQL_Agent(3, 2)
agent.epsDecay = 1000
agent.gamma = 0.2 # small memory since each game last one 1 or 2 rounds
agent.batch_size = 64
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
        plt.plot(moving_average(losses,n=300))
        plt.xlabel("Step")
        plt.ylabel("Average Loss")
        plt.subplot(122)
        plt.plot(moving_average(episode_rewards, n=600))
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.pause(0.1)

# Check action of network for each state
# white = hit
# black = stick
plt.figure()
sums = np.arange(1,25)
cards = np.arange(1,11)
actions = np.zeros((sums.shape[0],cards.shape[0]))
for sum in sums:
    for card in cards:
        actions[sum-1, card-1] = int(agent.act((sum,card,False), False, env.action_space.n))
plt.imshow(actions, cmap='hot', interpolation='nearest')
plt.xlabel("Dealer's Card")
plt.ylabel("Sum")
plt.show()

# Have agent play
state = env.reset()
done = False
while(1):
    print("NEW GAME")
    while(done == False):
        print("Player's sum: " + str(state[0]))
        print("Dealer's card: " + str(state[1]))
        print("Player has an ace: " + str(state[2]))
        action = agent.act(state, False, env.action_space.n)
        state, reward, done, info = env.step(int(action))

        if (int(action) == 1):
            print("Hit")
        else:
            print("Stick")

        print("Player's sum: " + str(state[0]))
        print("Dealer's card: " + str(state[1]))
        print("Player has an ace: " + str(state[2]))

        if(done):
            print("Reward " + str(reward))
    sleep(10)
    state = env.reset()
    done = False
env.close()
