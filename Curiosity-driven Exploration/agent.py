'''
inverse loss = cross entropy ( pred_action / real_action)
forward loss = mse(pred_next_feature, real_next_feature)
actor loss = -torch.min(surr,1-eps,1+eps)
critic loss = mse(v, td_target)
total loss = actor + 0.5 * critic + forward + inverse
there is a Beta in the paper but I don't know why there is no beta in implementation code

Agent has to calculate td_target, adv, pred_action, pred_feature from (s,a,r,s_prime,done_mask)
'''
# TODO : tensorwatch로 visualization
from env import *
from model import *
from utils import *
import config as cf
import torch
import random
import numpy as np
from torch.distributions.categorical import Categorical
from collections import deque
from tqdm import tqdm
### agent
# TODO : agent class로 합치기?
#### make batch from memory
# actor_optimizer, critic_optimizer, icm_optimizer
def train_net(memory, actor, critic, icm,optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mse = nn.MSELoss()
    random.shuffle(memory)
    for n in tqdm(range(len(memory) // cf.batch_size)):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, prob_lst = [], [], [], [], [], []
        for transition in memory[n * cf.batch_size:(n + 1) * cf.batch_size]:
            s, a, r, s_prime, done, prob = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
            prob_lst.append([prob])

        s = torch.tensor(s_lst, dtype=torch.float).to(device)
        a = torch.tensor(a_lst).to(device)
        r = torch.tensor(r_lst, dtype=torch.float).to(device)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float).to(device)
        done = torch.tensor(done_lst, dtype=torch.float).to(device)
        prob = torch.tensor(prob_lst, dtype=torch.float).to(device)

        real_feature, pred_feature, pred_action = icm((s, a, s_prime))
        intrinsic_reward = (real_feature - pred_feature).pow(2).sum(-1).unsqueeze(-1)
        intrinsic_reward = normalization(intrinsic_reward)
        total_reward = r + intrinsic_reward
        pred_action = pred_action.gather(1,a)

        old_v = critic(s)
        td_target = total_reward + cf.gamma * critic(s_prime) * done
        delta = (td_target - old_v) * done
        adv = discounted_r(delta,gamma = cf.gamma, gae= True)
        adv = normalization(adv).to(device)
        # adv = adv.to(device)
        for _ in range(cf.epoch):
    #### calculate td_target, delta, adv
            new_v = critic(s)
            pi = actor(s)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob))
            surr = ratio.squeeze() * adv
            clip = torch.clamp(surr, 1 - cf.eps, 1 + cf.eps)
            actor_loss = -torch.min(clip,surr).mean()
            critic_loss = mse(td_target.detach(),new_v)
            inv_loss = mse(pred_action, prob.detach())
            forward_loss = mse(pred_feature, real_feature)
            loss = actor_loss + 0.5 * critic_loss+ 0.5 * inv_loss + 0.5 * forward_loss
            # actor_optimizer.zero_grad()
            # loss.backward(retain_graph = True)
            # actor_optimizer.step()
            #
            # critic_optimizer.zero_grad()
            # loss.backward(retain_graph = True)
            # critic_optimizer.step()
            # #
            # icm_optimizer.zero_grad()
            # loss.backward(retain_graph = True)
            # icm_optimizer.step()
            # torch.cuda.empty_cache()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

