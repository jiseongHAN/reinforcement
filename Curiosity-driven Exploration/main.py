from env import *
from model import *
from utils import *
import config as cf
import torch
import random
import numpy as np
from torch.distributions.categorical import Categorical
from collections import deque
from agent import *


env_id = cf.env
env = MarioEnv(env_id, cf.stacked_frame, cf.height, cf.width, cf.skip)
n_action = env.n_action
Critic = CNNCritic()
Actor = CNNActor(n_action)
ICM = ICMModel(n_action)
memory = deque(maxlen= 10000)
optimizer = optim.Adam(list(Critic.parameters())+ list(Actor.parameters())+ list(ICM.parameters()),lr = cf.lr)
train = TRAIN(memory,Actor,Critic,ICM,optimizer)
### make transition or roll-out
for n_epi in range(cf.iter_max):
    s = env.reset()
    done_mask = 1
    score = 0.0
    step = 0
    while done_mask == 1:
        prob = Actor(torch.tensor(s,dtype=torch.float).unsqueeze(0))
        action = Categorical(prob)
        s, a, r, s_prime, done_mask = env.run(action.sample().item())
        score += r
        data = (s,a,r,s_prime,done_mask)
        s = s_prime
        memory.append(data)
        step += 1
        if done_mask == 0:
            break
    train()
    # TODO : train_net에 필요한 것 : intrinsic reward 구하기
    print('#{} 총 점수 : {} 스텝 : {}'.format(n_epi,score,step))
    score = 0.0
    step = 0.0
    train.memory = deque(maxlen=10000)

