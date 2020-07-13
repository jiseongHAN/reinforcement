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




def train(act, act_target, crt, crt_target, memory, batch_size, gamma, actor_optimizer, critic_optimizer):
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    y = torch.FloatTensor(r).unsqueeze(-1) + gamma * crt_target(s_prime,act_target(s_prime).squeeze()) * torch.FloatTensor(done).unsqueeze(-1)
    critic_loss = torch.mean((y - crt(s,a))**2)
    actor_loss = -torch.mean(crt(s, act(torch.FloatTensor(s) ).squeeze()  ))

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)



def main():
    gamma = 0.99
    batch_size = 64
    env = gym.make("Pendulum-v0")
    N = 10000
    eps = 0.001
    memory = replay_memory(N)
    epoch = 1000
    update_interval = 4
    crt = critic(env.observation_space.shape[0])
    crt_target = critic(env.observation_space.shape[0])
    act = actor(env.observation_space.shape[0], env.action_space.shape[0])
    act_target = actor(env.observation_space.shape[0], env.action_space.shape[0])
    critic_optimizer = optim.Adam(crt.parameters(), lr=0.0005)
    actor_optimizer = optim.Adam(act.parameters(), lr=0.0005)
    OU = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=float(1) * np.ones(1))

    t = 0
    for k in range(epoch):
        s = env.reset()
        done = False
        total_score = 0

        while not done:
            if eps > np.random.rand():
                a = env.action_space.sample()
            else:
                a = np.clip(act(s).detach().numpy()+ OU(),env.action_space.low, env.action_space.high)
            s_prime, r, done, _ = env.step(a)
            memory.push((list(s), float(r), int(a), list(s_prime), int(1 - done)))
            s = s_prime
            total_score += r
            if len(memory) > 2000:
                train(act, act_target, crt, crt_target, memory, batch_size, gamma, actor_optimizer,critic_optimizer)
                t += 1
            if t % update_interval == 0:
                soft_copy_weights(crt, crt_target,0.01)
                soft_copy_weights(act, act_target, 0.01)
        print("Epoch : %d | score : %f" % (k, total_score))



if __name__ ==  "main":
    main()
