'''
inverse loss = cross entropy ( pred_action / real_action)
forward loss = mse(pred_next_feature, real_next_feature)
actor loss = -torch.min(surr,1-eps,1+eps)
critic loss = mse(v, td_target)
total loss = actor + 0.5 * critic + forward + inverse
there is a Beta in the paper but I don't know why there is no beta in implementation code

Agent has to calculate td_target, adv, pred_action, pred_feature from (s,a,r,s_prime,done_mask)
'''

from env import *
from model import *
from utils import *
import config as cf
import torch
import random
import numpy as np
from torch.distributions.categorical import Categorical
from collections import deque


#### to main
# env_id = cf.env
# env = MarioEnv(env_id, cf.stacked_frame, cf.height, cf.width, cf.skip)
# n_action = env.n_action
# Critic = CNNCritic()
# Actor = CNNActor(n_action)
# ICM = ICMModel(n_action)
# memory = deque(maxlen= 10000)
#



### agent
# TODO : agent class로 합치기?
#### make batch from memory
class TRAIN():
    def __init__(self,memory,actor,critic,icm,optimizer):
        self.memory = memory
        self.actor = actor
        self.critic = critic
        self.icm = icm
        self.optimizer = optimizer
    def __call__(self):
        self.train_net()

    def train_net(self):
        mse = nn.MSELoss()
        memory = list(self.memory)
        random.shuffle(memory)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for data in memory:
            s, a, r, s_prime, done_mask = data
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s = torch.tensor(s_lst,dtype = torch.float)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst, dtype = torch.float)
        s_prime = torch.tensor(s_prime_lst, dtype = torch.float)
        done_mask = torch.tensor(done_mask_lst, dtype = torch.float)

        old_p = self.actor(s)
        old_policy = old_p.gather(1,a)
        real_feature, pred_feature, pred_action = self.icm((s,a,s_prime))
        intrinsic_reward  = (real_feature - pred_feature).pow(2).sum(-1).unsqueeze(-1)
        intrinsic_reward = normalization(intrinsic_reward)
        total_reward = r + intrinsic_reward
        real_feature = normalization(real_feature)
        pred_feature = normalization(pred_feature)

        #### calculate td_target, delta, adv
        for _ in range(cf.epoch):
            v_prime = self.critic(s_prime)
            v = self.critic(s)
            td_target = total_reward + cf.gamma * v_prime * done_mask
            delta = (td_target - v) * done_mask
            adv = discounted_r(delta, gamma=cf.gamma, gae= True)
            adv = normalization(adv) # TODO : adv 계산 더 빨리 하는 방법
            pi = self.actor(s)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(old_policy))
            surr = ratio.squeeze() * adv
            clip = torch.clamp(surr, 1 - cf.eps, 1 + cf.eps)
            actor_loss = -torch.min(clip,surr).mean()
            critic_loss = mse(td_target.detach(),v)
            inv_loss = mse(pred_action,old_p)
            forward_loss = mse(pred_feature, real_feature)
            loss = actor_loss + 0.5 * critic_loss + inv_loss + forward_loss

            self.optimizer.zero_grad()
            loss.backward(retain_graph = True)
            self.optimizer.step()
