import torch
import random
from collections import deque
import cv2
import numpy as np

def prepro(img):
    ret = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return ret/255.

class ReplayBuffer():
    def __init__(self,len,batch_size):
        self.buffer = deque(maxlen=len)
        self.batch_size = batch_size

    def add(self,data):
        self.buffer.append(data)

    def sample(self):
        mini_batch = random.sample(self.buffer,self.batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_lst.append(done)
        s = torch.tensor(s_lst, dtype=torch.float)
        a = torch.tensor(a_lst, dtype=torch.float)
        r = torch.tensor(r_lst, dtype=torch.float)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
        done = torch.tensor(done_lst, dtype=torch.float)
        return s, a, r, s_prime, done

    def __call__(self, *args, **kwargs):
        return len(self.buffer)



# TODO : normalization 함수 만들기 / Ornstein-Uhlenbeck process로 노이즈 생성 함수
