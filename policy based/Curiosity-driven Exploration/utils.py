import os
import numpy as np
import torch
import config as cf
import torch.nn.functional as F
import cv2

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

def prepro(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state,(cf.height,cf.width)) / 255.0
    return state


def get_icm_loss(states, next_states, actions, action_probs,icm): # actions -> a 인듯?
    action_pred, phi2_pred, phi1, phi2 = icm(states, next_states, action_probs)
    inverse_loss = F.cross_entropy(action_pred, actions.view(-1))
    forward_loss = 0.5 * F.mse_loss(phi2_pred, phi2.detach(), reduction = 'none').sum(-1).mean()
    intrinsic_reward = 0.5 * F.mse_loss(phi2_pred, phi2.detach(), reduction = 'none').sum(-1)
    intrinsic_reward = normalization(intrinsic_reward)
    return inverse_loss, forward_loss, intrinsic_reward


def model_save(model, model_path, name):
    try:
        if os.path.isdir(model_path):
            torch.save(model.state_dict(), model_path+ '/{}.pth'.format(str(name)))
        else:
            os.mkdir(model_path)
            torch.save(model.state_dict(), model_path + '/{}.pth'.format(str(name)))
    except:
        print('Save failed')

        #print(os.path.exists("/home/el/myfile.txt"))
