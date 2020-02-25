## Authors: Artur Filipowicz
## Version: 0.9
## Copyright: 2020
## MIT License

import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import math, random
from collections import deque

class DQL_Agent(nn.Module):
    def __init__(self, stateSize, numActions):
        super(DQL_Agent, self).__init__()
        self.lifetime = 0
        # exploration parameters
        self.epsInit = 1.0
        self.epsFinal = 0.01
        self.epsDecay = 5000
        # learning parameters
        self.batch_size = 32
        self.memory = deque(maxlen=self.batch_size*4)
        self.optimizer = []
        self.gamma = 0.99
        self.updateRefNetFreq = self.batch_size*1

        self.net = nn.Sequential(
            nn.Linear(stateSize, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, numActions),
        )

        self.ref = copy.deepcopy(self.net)

    def changeNet(self, net):
        self.net = net
        self.ref = copy.deepcopy(self.net)

    def memorize(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.memory.append((state, action, reward, next_state, done))

    def recall(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def study(self):
        state, action, reward, next_state, done = self.recall(self.batch_size)

        state = Variable(torch.FloatTensor(state))
        next_state = Variable(torch.FloatTensor(next_state))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        Q_t = self.net(state) # raw Q values from network
        Q_A_t = Q_t.gather(1, action.unsqueeze(1)).squeeze(1) # Q values of taken actions

        Q_t1 = self.ref(next_state) # raw Q values from reference network
        Q_max_t1 = Q_t1.max(1)[0] # max Q value of next state
        loss = (Q_A_t - (reward + self.gamma*Q_max_t1*(1-done))).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.lifetime % self.updateRefNetFreq == 0:
            self.ref.load_state_dict(self.net.state_dict())

        return loss

    def act(self, state, explore, action_space):

        eps = self.epsFinal+(self.epsInit-self.epsFinal)*math.exp(-1.*self.lifetime/self.epsDecay)

        if random.random() < eps and explore == True:
            action = random.randrange(action_space)
        else:
            state = Variable(torch.FloatTensor([state]))
            Q = self.net(state)
            action = Q.argmax()
        return action