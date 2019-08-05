import os
import numpy as np
import torch
import config as cf
import torch.nn.functional as F
import cv2
from torch._six import inf


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


def normalization(state,clip):
    if type(state) == torch.Tensor:
        mean = state.mean()
        std = state.std()
        state = (state- mean) / (std + 1e-12)
        if clip:
            state = torch.clamp(state,-clip,clip)
    else:
        mean = np.mean(state)
        std = np.std(state)
        state = (state-mean) / (std + 1e-12)
        if clip:
            state = np.clip(state,-clip,clip)
    return state


def prepro(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # state = cv2.resize(state,(cf.height,cf.width)) / 255.0
    state = cv2.resize(state,(cf.height,cf.width))

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


def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm

def make_ma(data,term):
    ma = []
    for i in range(len(data)//term):
        ma.append(sum(data[i*term:(i+1)*term])/term)
    return ma
