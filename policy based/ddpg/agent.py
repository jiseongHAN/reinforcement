# TODO : transition 이용해서 actor-critic 업데이트하는 함수 만들기 -> real agent 만들기
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from copy import copy
from model import *
import torch.optim as optim

class DDPG():
    def __init__(self,memory,
                 n_action,
                 device,
                 tau=cf.tau,
                 a_lr=cf.a_lr,
                 c_lr=cf.c_lr,
                 gamma=cf.gamma):
        self.memory = memory
        self.tau = tau
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.gamma = gamma
        self.q = CNNCritic().to(device)
        self.q_target = CNNCritic().to(device)
        self.actor = CNNActor(n_action).to(device)
        self.actor_target = CNNActor(n_action).to(device)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr = c_lr)
        self.a_optimizer = optim.Adam(self.actor.parameters(), lr = a_lr)
        self.device = device
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(n_action))
        self.mse = nn.MSELoss()

    def get_action(self,state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = normalization(state)
        action = self.actor(state.unsqueeze(0))
        try:
            return action.squeeze().cpu().data.numpy() + self.ou_noise()
        except:
            return action.squeeze().data.numpy() + self.ou_noise()

    def train(self):
        s, a, r, s_prime, done = self.memory.sample()
        s, s_prime  = normalization(s), normalization(s_prime)
        # r = normalization(r)
        target = r + self.gamma * self.q_target(s_prime,self.actor_target(s_prime)) * done
        q_loss = self.mse(self.q(s,a), target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        a_loss = -self.q(s,self.actor(s)).mean()
        self.a_optimizer.zero_grad()
        a_loss.backward()
        self.a_optimizer.step()
        self.soft_update(self.q,self.q_target)
        self.soft_update(self.actor,self.actor_target)
        return q_loss, a_loss

    def soft_update(self,net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
