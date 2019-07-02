import torch.optim as optim
from model import *
import torch.nn as nn
import torch
from collections import deque
import hp
import random

##s a r s' done

def train(actor, critic, data, actor_optim, critic_optim):
    s_lst, a_lst, r_lst, s_prime_lst, done_lst = [],[],[],[],[]
    random.shuffle(data)
    data = list(data)
    for transition in data:
        s, a, r, s_prime, done = transition
        s_lst.append(s)
        a_lst.append(a)
        r_lst.append([r])
        s_prime_lst.append(s_prime)
        done_lst.append([done])

    s = torch.tensor(s_lst, dtype=torch.float)
    a = torch.tensor(a_lst, dtype=torch.float)
    r = torch.tensor(r_lst, dtype=torch.float)
    s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
    done = torch.tensor(done_lst, dtype=torch.float)

    mu, std, logstd = actor(s)
    old_p = log_density(a, mu, std, logstd)

    for n in range(len(data) // hp.batch_size):
        s=s[n*hp.batch_size:(n+1)*hp.batch_size]
        a=a[n*hp.batch_size:(n+1)*hp.batch_size]
        r=r[n*hp.batch_size:(n+1)*hp.batch_size]
        s_prime=s_prime[n*hp.batch_size:(n+1)*hp.batch_size]
        done=done[n*hp.batch_size:(n+1)*hp.batch_size]

        for _ in range(hp.epoch):
            td_target = r + hp.gamma * critic(s_prime) * done
            delta = td_target - critic(s)
            adv = discounted_r(delta, gamma=hp.gamma, gae=True,lam=hp.lam)
            adv = torch.tensor(adv)
            adv = (adv - adv.mean()) / (adv.std() + 1e-12)
            new_mu, new_std, new_log = actor(s)
            new_p = log_density(a,new_mu,new_std,new_log)
            ratio = new_p / old_p
            surr = ratio * adv

            a_loss = torch.clamp(surr,1-hp.eps,1+hp.eps) * adv
            c_loss = F.smooth_l1_loss(td_target, critic(s))
            loss = a_loss + 0.5 * c_loss
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            loss.mean().backward(retain_graph=True)
            actor_optim.step()
            critic_optim.step()







