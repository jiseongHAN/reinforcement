'''
inverse loss = cross entropy ( pred_action / real_action)
forward loss = mse(pred_next_feature, real_next_feature)
actor loss = -torch.min(surr,1-eps,1+eps)
critic loss = mse(v, td_target)
total loss = actor + 0.5 * critic + forward + inverse
there is a Beta in the paper but I don't know why there is no beta in implementation code
'''

from model import *
import config as cf
import gym
from env import *
import torch
import torch.nn as nn
import torch.optim as optim

