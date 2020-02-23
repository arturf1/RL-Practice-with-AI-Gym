import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from DQL import DQL_Agent

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

env = gym.make('MountainCar-v0')
print(env.observation_space)
print(env.action_space)
agent = DQL_Agent(2, 3)
agent.optimizer = torch.optim.Adam(agent.parameters())

steps = 20000
losses = []
episode_reward = 0
episode_rewards = []
avg_loss = 0
avg_losses = []

state = env.reset()
episode_reward = 0
for step in range(1, steps + 1):

    agent.lifetime += 1
    action = agent.act(state, True, env.action_space.n)
    next_state, reward, done, _ = env.step(int(action))
    print(100*(reward==1))
    reward = 100*(state[0] > -0.8)*(state[0]+0.8)+100*(reward==1)
    print(reward)
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
        avg_losses.append((avg_loss + loss.data.numpy()) / step)
        avg_loss = 0

    try:
        avg_loss = avg_loss + loss.data.numpy()
    except:
        pass

    if step % 500 == 0:
        plt.cla()
        plt.subplot(121)
        plt.plot(avg_losses)
        plt.subplot(122)
        plt.plot(episode_rewards)
        plt.pause(0.1)

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


