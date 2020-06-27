import torch.optim as optim
from model import *
import torch.nn as nn
import torch
from collections import deque
import hp
import random
import numpy as np
from tqdm import tqdm

##s a r s' done

def train(actor, critic, buffer, actor_optim, critic_optim):
    mse = nn.MSELoss()
    s_lst, a_lst, r_lst, s_prime_lst, done_lst = [],[],[],[],[]
    idx = np.arange(len(buffer))
    random.shuffle(idx)
    loss_lst = []
    for transition in buffer:
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
    old_v =  critic(s)
    old_p = log_density(a, mu, std, logstd).detach()
    td_target = r + hp.gamma * critic(s_prime) * done
    delta = td_target - old_v
    delta_d = delta * done
    adv = discounted_r(delta_d, gamma=hp.gamma, gae=True)
    adv = torch.tensor(adv)
    adv = (adv - adv.mean()) / (adv.std() + 1e-12)
    # returns = discounted_r(r,gamma=hp.gamma,gae=False)
    # returns = torch.tensor(returns).unsqueeze(-1)
    for n in range(len(buffer) // hp.batch_size):
        s_b=s[idx[n*hp.batch_size:(n+1)*hp.batch_size]]
        a_b=a[idx[n*hp.batch_size:(n+1)*hp.batch_size]]
        old_b = old_p[idx[n*hp.batch_size:(n+1)*hp.batch_size]]
        td_b = td_target[idx[n*hp.batch_size:(n+1)*hp.batch_size]]
        adv_b = adv[idx[n*hp.batch_size:(n+1)*hp.batch_size]]
        # ret_b = returns[idx[n*hp.batch_size:(n+1)*hp.batch_size]]
        # old_v_b = old_v[idx[n*hp.batch_size:(n+1)*hp.batch_size]]
        for _ in range(hp.epoch):
            new_v = critic(s_b)
            new_mu, new_std, new_log = actor(s_b)
            new_p = log_density(a_b,new_mu,new_std,new_log)
            ratio = torch.exp(new_p - old_b)
            surr = ratio.squeeze() * adv_b
            clip = torch.clamp(surr,1-hp.eps,1+hp.eps) * adv_b
            a_loss = -torch.min(clip,surr).mean()
            critic_loss = mse(td_b.detach(),new_v)
            loss = a_loss + critic_loss

            actor_optim.zero_grad()
            loss.backward(retain_graph=True)
            actor_optim.step()

            critic_optim.zero_grad()
            loss.backward(retain_graph=True)
            critic_optim.step()

            loss_lst.append(loss.data.numpy())
    print('\n Loss : {}'.format(np.mean(loss_lst)))
    return np.mean(loss_lst)

#TODO : checking matrix dimension
#TODO : update gradient per model
#TODO : edit critic loss according to openai Baseline


