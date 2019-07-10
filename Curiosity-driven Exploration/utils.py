import numpy as np
import torch
import config as cf

def discounted_r(data,gamma,gae:bool):
    R = 0
    r_lst = []
    if gae:
        for r in reversed(data):
            R = r + cf.lam * gamma * R
            r_lst.append(R)
        r_lst.reverse()
    else:
        for r in reversed(data):
            R = r + gamma * R
            r_lst.append(R)
        r_lst.reverse()
    return torch.tensor(r_lst)


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
