from agent import *
from env import *
from model import *
from utils import *
import config as cf
import torch
import random
import numpy as np
from torch.distributions.categorical import Categorical
from collections import deque
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


env_id = cf.env
mario = gym_super_mario_bros.make(env_id)
env = SkippedEnv(mario, cf.skip)
# env = JoypadSpace(env,COMPLEX_MOVEMENT)
env = JoypadSpace(env,RIGHT_ONLY)
n_action = env.action_space.n
critic = CNNCritic().to(device)
actor = CNNActor(n_action).to(device)
icm = ICMModel(n_action).to(device)
memory = []
actor_optimizer = optim.Adam(actor.parameters(),lr = cf.lr)
critic_optimizer = optim.Adam(critic.parameters(),lr = cf.lr)
icm_optimizer = optim.Adam(icm.parameters(),lr = cf.lr)
optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()) + list(icm.parameters()), lr = cf.lr)

### make transition or roll-out
for n_epi in range(cf.iter_max):
    s = env.reset()
    s = prepro(s)
    state = deque(maxlen=cf.stacked_frame)
    for _ in range(cf.stacked_frame):
        state.append(s)
    score = 0.0
    step = 0
    done = False
    while not done:
        # env.render()
        prob = actor(torch.tensor(state,dtype=torch.float).unsqueeze(0).to(device))
        action = Categorical(prob.cpu())
        a = action.sample().item()
        s_prime, r, done, info = env.step(a)
        s_prime = prepro(s_prime)
        state_prime = state.copy()
        state_prime.append(s_prime)
        score += r
        done_mask = 0 if done else 1
        data = (state,a,r,state_prime,done_mask, prob.squeeze()[a].item())
        state = state_prime
        memory.append(data)
        step += 1
        if done_mask == 0:
            break
        if len(memory) == 5120:
            train_net(memory,actor,critic,icm,optimizer)
            memory = []
    print('#{} 총 점수 : {} 스텝 : {}'.format(n_epi,score,step))
    score = 0.0
    step = 0.0

# actor_optimizer,critic_optimizer,icm_optimizer

## TODO : 자잘한 버그 / agent와 main 간의 호환! - env 안정성
## TODO : GPU 버전 만들기 / + a3c
