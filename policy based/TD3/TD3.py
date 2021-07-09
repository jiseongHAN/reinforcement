import numpy as np
import gym
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim
import random
import torch.nn.functional as F

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


class critic(nn.Module):
    def __init__(self,n_obs):
        super(critic,self).__init__()
        self.layer_q = nn.Linear(n_obs,64)
        self.layer_a = nn.Linear(1,64)
        self.layer2 = nn.Linear(128,128)
        self.q = nn.Linear(128,1)

        self.seq = nn.Sequential(
            self.layer_q,
            self.layer_a,
            self.layer2,
            self.q
        )

        self.seq.apply(init_weights)

    def forward(self,x,a):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x)
        if type(a) != torch.Tensor:
            a = torch.FloatTensor(a).unsqueeze(1)

        x = F.relu(self.layer_q(x))
        a = F.relu(self.layer_a(a))
        concat = torch.cat((x,a),dim=1)
        q = F.relu(self.layer2(concat))
        return self.q(q)



class actor(nn.Module):
    def __init__(self,n_obs,n_action):
        super(actor,self).__init__()
        self.layer1 = nn.Linear(n_obs,128)
        self.layer2 = nn.Linear(128,128)
        self.mu = nn.Linear(128,n_action)

        self.seq = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU(),
            self.mu,
            nn.Tanh()
        )

        self.seq.apply(init_weights)

    def forward(self,x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x)
            return 2 * self.seq(x)
        else:
            return 2 * self.seq(x)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def soft_copy_weights(net,net_target,tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def cliped_noise(sigma,c=3):
    noise = np.random.normal(0, sigma)
    return np.clip(noise,-c,c)


def train(act, act_target, crt1, crt2, crt1_target, crt2_target, memory, batch_size, gamma, actor_optimizer, critic1_optimizer,critic2_optimizer):
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))

    with torch.no_grad():
        a_tilde = torch.clamp(act_target(s_prime)+ cliped_noise(sigma=0.1),-2,2)

    r = torch.FloatTensor(r).unsqueeze(-1)
    done = torch.FloatTensor(done).unsqueeze(-1)

    with torch.no_grad():
        y1 = r + gamma * crt1_target(s_prime,a_tilde) * done
        y2 = r + gamma * crt2_target(s_prime,a_tilde) * done

    y = torch.min(y1,y2)

    q1 = crt1(s,a)
    q2 = crt2(s,a)

    critic1_loss = F.smooth_l1_loss(q1, y).mean()
    critic2_loss = F.smooth_l1_loss(q2, y).mean()

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    critic2_optimizer.step()

    actor_loss = -crt1(s,act(s)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return float(actor_loss)


def main():
    env = gym.make("Pendulum-v0")
    gamma = 0.99
    batch_size = 32
    N = 50000
    memory = replay_memory(N)
    epoch = 1000
    sigma = 0.1
    crt1 = critic(env.observation_space.shape[0])
    crt1_target = critic(env.observation_space.shape[0])
    crt1_target.load_state_dict(crt1.state_dict())

    crt2 = critic(env.observation_space.shape[0])
    crt2_target = critic(env.observation_space.shape[0])
    crt2_target.load_state_dict(crt2.state_dict())


    act = actor(env.observation_space.shape[0], env.action_space.shape[0])
    act_target = actor(env.observation_space.shape[0], env.action_space.shape[0])
    act_target.load_state_dict(act.state_dict())

    critic1_optimizer = optim.Adam(crt1.parameters(), lr=0.0005)
    critic2_optimizer = optim.Adam(crt2.parameters(), lr=0.0005)
    actor_optimizer = optim.Adam(act.parameters(), lr=0.001)
    loss = 0.0
    for k in range(epoch):
        s = env.reset()
        done = False
        total_score = 0
        while not done:
            a = act(s).detach().numpy()+ np.random.normal(0, 0.1)
            s_prime, r, done, _ = env.step(a)
            memory.push((list(s), float(r), int(a), list(s_prime), int(1 - done)))
            s = s_prime
            total_score += r
            if len(memory) > 2000:
                loss = train(act, act_target, crt1, crt2, crt1_target, crt2_target, memory, batch_size, gamma, actor_optimizer, critic1_optimizer,critic2_optimizer)
                soft_copy_weights(crt1, crt1_target,0.005)
                soft_copy_weights(crt2, crt2_target,0.005)
                soft_copy_weights(act, act_target, 0.005)
        print("Epoch : %d | score : %f | loss : %4f" % (k, total_score, loss))



if __name__ ==  "__main__":
    main()
