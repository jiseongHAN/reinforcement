import numpy as np
import gym
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim
import random
import torch.nn.functional as F


env = gym.make('CartPole-v1').unwrapped

'''
initialize replay memory D with N
'''

class replay_memory(object):
    def __init__(self,N):
        self.memory = deque(maxlen=N)


    def push(self,transition):
        self.memory.append(transition)

    def sample(self,n):
        return random.sample(self.memory,n)


    def __len__(self):
        return len(self.memory)


class mlp(nn.Module):
    def __init__(self,n_obs,n_action):
        super(mlp,self).__init__()
        self.layer1 = nn.Linear(n_obs,128)
        self.layer2 = nn.Linear(128,128)
        self.q = nn.Linear(128,n_action)

        self.seq = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU(),
            self.q
        )

        self.seq.apply(init_weights)

    def forward(self,x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x)
            return self.seq(x)
        else:
            return self.seq(x)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(q, memory, batch_size, gamma, optimizer):
    s,r,a,s_prime,done = list(map(list, zip(*memory.sample(batch_size))))
    y = torch.FloatTensor(r) + gamma*q(s_prime).max(1)[0]*torch.FloatTensor(done)
    a = torch.tensor(a).unsqueeze(-1)
    y = y.unsqueeze(-1)
    loss = torch.sum((y - q(s).gather(1,a))**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    gamma = 0.99
    batch_size = 32
    env = gym.make("CartPole-v1")
    N = 10000
    eps = 0.001
    memory = replay_memory(N)
    epoch = 1000
    q = mlp(env.observation_space.shape[0],env.action_space.n)
    optimizer = optim.Adam(q.parameters(),lr=0.0005)
    for k in range(epoch):
        s = env.reset()
        done = False
        total_score = 0

        while not done:
            if eps > np.random.rand():
                a = env.action_space.sample()
            else:
                a = np.argmax(q(s).detach().numpy())
            s_prime, r, done, _ = env.step(a)
            memory.push((list(s),float(r),int(a),list(s_prime),int(1-done)))
            s = s_prime
            total_score += r
            if len(memory) > 2000:
                train(q,memory,batch_size,gamma,optimizer)
        print("Epoch : %d | score : %f" %(k, total_score))



if __name__ ==  "__main__":
    main()


