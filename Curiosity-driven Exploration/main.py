from agent import *
from env import *
from model import *
from utils import *
import config as cf
import torch
import random
import numpy as np
from torch.distributions.categorical import Categorical
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from tensorboardX import SummaryWriter



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_id = cf.env
env = env_make(env_id,SIMPLE_MOVEMENT)
n_action = env.action_space.n
critic = CNNCritic().to(device)
actor = CNNActor(n_action).to(device)
icm = ICMModel(n_action).to(device)
memory = []
actor_optimizer = optim.Adam(actor.parameters(),lr = 0.0001)
critic_optimizer = optim.Adam(critic.parameters(),lr = 0.0002)
icm_optimizer = optim.Adam(icm.parameters(),lr = 0.001)
optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()) + list(icm.parameters()), lr = cf.lr)

### make transition or roll-out
for n_epi in range(cf.iter_max):
    s = env.reset()
    s = prepro(s)
    state = np.zeros((cf.stacked_frame,cf.height,cf.width))
    for i in range(cf.stacked_frame):
        state[i,::] = s
    score = 0.0
    step = 0
    n_train = 0
    done = False
    while not done:
        # env.render()
        prob = actor(torch.tensor(state,dtype=torch.float).unsqueeze(0).to(device))
        action = Categorical(prob.cpu())
        a = action.sample().item()
        s_prime, r, done, info = env.step(a)
        state_prime = np.zeros_like(state)
        state_prime[:3] = state[1:]
        state_prime[3, :, :] = prepro(s_prime)
        score += r
        done_mask = 0 if done else 1
        data = (state,a,r,state_prime,done_mask, prob.squeeze()[a].item())
        # print('state = {}'.format((state == state_prime).all()))
        # print('fixed = {}'.format((s == prepro(s_prime)).all()))
        memory.append(data)
        state = state_prime
        step += 1
        if done:
            break
        if len(memory) >= 1600:
            n_train += 1
            train_net(memory,actor,critic,icm,optimizer)
            memory = []
    print('#{} 총 점수 : {} 스텝 : {}'.format(n_epi,score,step))
    if info['stage'] == 2:
        print('Stage 1 Clear')
    score = 0.0
    step = 0.0
env.close()
# actor_optimizer,critic_optimizer,icm_optimizer

## TODO : 자잘한 버그 / agent와 main 간의 호환! - env 안정성( 어느정도 클리어 )
## TODO : GPU 버전 만들기(O) / + a3c
## TODO : ICM 넣어서 돌려보기
