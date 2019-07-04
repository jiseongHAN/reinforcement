import torch.optim as optm
import torch
from model import *
from train_ppo import *
import gym
from collections import deque
import hp
import numpy as np
import math
env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('MountainCarContinuous-v0')


action_dim = env.action_space.shape[0]
input_dim = env.observation_space.shape[0]
actor = Actor(input_dim,action_dim)
buffer = []
critic = Critic(input_dim)
actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                          weight_decay=hp.l2_rate)
n_train = 0
loss_lst = []
for _ in range(hp.iter_max):
    s = env.reset()
    done = False
    score = 0.0
    step =0
    while not done:
        env.render()
        # s = norm_state(s)
        mu, std, logstd = actor(torch.from_numpy(s).float())
        a = get_action(mu,std)
        s_prime, r, done, _ = env.step(torch.clamp(torch.Tensor(a),-1,1).data.numpy())
        score += r
        r = np.clip(r,-5,5)
        done_mask = 0 if done else 1
        # s_prime = norm_state(s_prime)
        sarsd = (s,a,r,s_prime, done_mask)
        buffer.append(sarsd)
        step += 1
        s = s_prime
        if done:
            break
    if len(buffer) > 2000:
        loss = train(actor,critic,buffer,actor_optim,critic_optim)
        loss_lst.append(loss)
        n_train += 1
        buffer = []
    if score > 200.0:
        print('Clear!!')
        break
    print('#{} : 총 점수 : {} 총 스텝 : {}'.format(n_train,score,step))
    score = 0.0
    step = 0
env.close()
