'''
action_sapce box(3,)
-> [(-1,1),(0,1),(0,1)]

observation dim -> (96,96,3)

'''
from ipywidgets import interactive
from utils import *
import gym
import config as cf
import cv2
import torch
from agent import *
import matplotlib.pyplot as plt

env = gym.make(cf.env)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = DDPG(memory = ReplayBuffer(cf.len,cf.batch_size,device), n_action = env.action_space.shape[0],device = device)

n_epi = 0
while True:
    s = env.reset()
    s = prepro(s)
    done = False
    score = 0.0
    while not done:
        # env.render()
        a = agent.get_action(s)
        if agent.memory() > 20000:
            s_prime, r, done, info = env.step(a)
            # print(a)
        else:
            s_prime, r, done, info = env.step(env.action_space.sample())
        score += r
        s_prime = prepro(s_prime)
        done_mask = 0 if done else 1
        transition = (s,a,r,s_prime,done_mask)
        agent.memory.add(transition)
        s = s_prime
    n_epi += 1
    if agent.memory() > 20000:
        q_loss, a_loss = [], []
        for i in range(cf.epoch):
            ql, al = agent.train()
            q_loss.append(ql.cpu().data.numpy())
            a_loss.append(al.cpu().data.numpy())

        print('epi #{} | Q_loss : {} | A_loss : {}'.format(n_epi,sum(q_loss)/(len(q_loss)+1e-12),sum(a_loss)/(len(a_loss)+1e-12)))
    print('epi #{} | SCORE : {} '.format(n_epi,score))
env.close()


# TODO : transition 저장하는 함수 코드 및 트레인
