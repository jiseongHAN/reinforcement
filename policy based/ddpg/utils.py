import torch
import random
from collections import deque
import cv2
import numpy as np

def prepro(img):
    ret = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) / 255.
    ret = np.expand_dims(ret,0)
    return ret

class ReplayBuffer():
    def __init__(self,len,batch_size,device):
        self.buffer = deque(maxlen=len)
        self.batch_size = batch_size
        self.device = device
    def add(self,data):
        self.buffer.append(data)

    def sample(self):
        mini_batch = random.sample(self.buffer,self.batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
        s = torch.tensor(s_lst, dtype=torch.float).to(self.device)
        a = torch.tensor(a_lst, dtype=torch.float).to(self.device)
        r = torch.tensor(r_lst, dtype=torch.float).to(self.device)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float).to(self.device)
        done = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        return s, a, r, s_prime, done

    def __call__(self, *args, **kwargs):
        return len(self.buffer)

'''
code from https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
'''
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.0, 0.00, 0.0
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def normalization(state):
    if type(state) == torch.Tensor:
        mean = state.mean()
        std = state.std()
        state = (state- mean) / (std + 1e-12)
    else:
        mean = np.mean(state)
        std = np.std(state)
        state = (state-mean) / (std + 1e-12)
    return state


# TODO : normalization 함수 만들기 / Ornstein-Uhlenbeck process로 노이즈 생성 함수
