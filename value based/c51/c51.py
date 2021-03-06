import math
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
    def __init__(self,n_obs,n_action,n_atom,V_min,V_max):
        super(mlp,self).__init__()
        self.layer1 = nn.Linear(n_obs,128)
        self.layer2 = nn.Linear(128,128)
        self.q = nn.Linear(128,n_action*n_atom)
        self.n_action = n_action
        self.n_atom = n_atom
        self.seq = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU(),
            self.q
        )
        self.d_z = (V_max - V_min) / (n_atom-1)
        self.z = [V_min + i * self.d_z for i in range(n_atom)]
        self.V_min = V_min
        self.V_max = V_max

        self.seq.apply(init_weights)

    def forward(self,x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x)
        x = self.seq(x)
        x = x.view(-1, self.n_action, self.n_atom, )
        x = F.softmax(x,-1)
        return x

    def get_action(self,pi):
        q = (pi*torch.tensor(self.z)).sum(-1)
        return torch.argmax(q,-1,True)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(q, memory, batch_size, gamma, optimizer):
    s,r,a,s_prime,done = list(map(list, zip(*memory.sample(batch_size))))
    m = torch.zeros(batch_size, q.n_action , q.n_atom)
    p_prime = q(s_prime)
    a_max = q.get_action(p_prime)



    for i in range(batch_size):
        if done[i] == 0:
            tz = min(max(q.V_min,r[i]),q.V_max)
            bj = (tz - q.V_min) / q.d_z
            l,u = math.floor(bj),  math.ceil(bj)
            m[i][a[i]][int(l)] += (u -bj)
            m[i][a[i]][int(u)] += (bj-l)
        else:
            for j in range(q.n_atom):
                tz = min(max(q.V_min,r[i] + gamma*q.z[j]),q.V_max)
                bj = (tz - q.V_min) / q.d_z
                l,u = math.floor(bj), math.ceil(bj)
                m[i][a[i]][int(l)] += p_prime[i][a_max[i].squeeze()][j]*(u -bj)
                m[i][a[i]][int(u)] += p_prime[i][a_max[i].squeeze()][j]*(bj-l)

    # loss = -torch.sum(m*torch.log(q(s)))
    loss = F.binary_cross_entropy_with_logits(q(s),m)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def copy_weights(q,q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)

def main():
    gamma = 0.99
    batch_size = 32
    env = gym.make("CartPole-v1")
    N = 10000
    eps = 0.001
    memory = replay_memory(N)
    epoch = 1000
    n_atom = 51
    V_min = -3
    V_max = 5
    # update_interval = 4
    q = mlp(env.observation_space.shape[0],env.action_space.n,n_atom,V_min,V_max)
    # q_target = mlp(env.observation_space.shape[0],env.action_space.n,51,0,10)
    optimizer = optim.Adam(q.parameters(),lr=0.0005)
    for k in range(epoch):
        s = env.reset()
        done = False
        total_score = 0

        while not done:
            if eps > np.random.rand():
                a = env.action_space.sample()
            else:
                a = int(q.get_action(q(s)).detach().numpy())
            s_prime, r, done, _ = env.step(a)
            memory.push((list(s),float(r),int(a),list(s_prime),int(1-done)))
            s = s_prime
            total_score += r
            if len(memory) > 2000:
                train(q,memory,batch_size,gamma,optimizer)
                # print(loss)
                # t += 1
            # if t % update_interval == 0:
            #     copy_weights(q,q_target)
        print("Epoch : %d | score : %f" %(k, total_score))



if __name__ ==  "__main__":
    main()


