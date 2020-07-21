from atari.wrappers import *
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim
import random

'''
initialize replay memory D with N
'''

def arange(s):
    if not type(s) == 'numpy.ndarray':
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret,0)

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
    def __init__(self,n_frame,n_action):
        super(mlp,self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32,64, 3,1)
        self.fc = nn.Linear(20736,512)
        self.q = nn.Linear(512, n_action)

        self.seq = nn.Sequential(
            self.layer1,
            self.layer2,
            self.fc,
            self.q
        )

        self.seq.apply(init_weights)

    def forward(self,x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1,20736)
        x = torch.relu(self.fc(x))
        x = self.q(x)
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(q, q_target, memory, batch_size, gamma, optimizer):
    s,r,a,s_prime,done = list(map(list, zip(*memory.sample(batch_size))))
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    a_max = q(s_prime).max(1)[1].unsqueeze(-1)
    y = torch.FloatTensor(r).unsqueeze(-1) + gamma*q_target(s_prime).gather(1,a_max)*torch.FloatTensor(done).unsqueeze(-1)
    a = torch.tensor(a).unsqueeze(-1)
    loss = torch.sum((y - q(s).gather(1,a))**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def copy_weights(q,q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)

def main():
    gamma = 0.99
    batch_size = 64
    n_frame = 4
    env = FrameStack(ScaledFloatFrame(WarpFrame(make_atari('BreakoutNoFrameskip-v0'))), n_frame)
    N = 10000
    eps = 0.001
    memory = replay_memory(N)
    epoch = 1000
    update_interval = 50
    q = mlp(n_frame,env.action_space.n)
    q_target = mlp(n_frame,env.action_space.n)
    optimizer = optim.Adam(q.parameters(),lr=0.0005)
    t = 0
    for k in range(epoch):
        s = arange(env.reset())
        done = False
        total_score = 0

        while not done:
            if eps > np.random.rand():
                a = env.action_space.sample()
            else:
                a = np.argmax(q(s).detach().numpy())
            s_prime, r, done, _ = env.step(a)
            s_prime = arange(s_prime)
            memory.push((s,float(r),int(a),s_prime,int(1-done)))
            s = s_prime
            total_score += r
            if len(memory) > 2000:
                train(q,q_target,memory,batch_size,gamma,optimizer)
                t += 1
            if t % update_interval == 0:
                copy_weights(q,q_target)
        print("Epoch : %d | score : %f" %(k, total_score))



if __name__ ==  "__main__":
    main()


