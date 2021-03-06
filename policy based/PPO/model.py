import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hp
import os
class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, num_outputs)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)



def discounted_r(data,gamma,gae:bool):
    R = 0
    r_lst = []
    if gae:
        for r in reversed(data):
            R = r + hp.lam * gamma * R
            r_lst.append(R)
        r_lst.reverse()
    else:
        for r in reversed(data):
            R = r + gamma * R
            r_lst.append(R)
        r_lst.reverse()
    return r_lst


def norm_state(state):
    mean = np.mean(state)
    std = np.std(state)
    state = (state-mean) / std
    return state

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

# TODO : replace activation function tanh -> swish : f(x) = x*sigmoid(x)
