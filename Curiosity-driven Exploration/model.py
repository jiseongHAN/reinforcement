import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
import numpy as np
import config as cf
def swish(x):
    ret = x * torch.sigmoid(x)
    return ret

class CNNActor(nn.Module):
    def __init__(self,n_action):
        super(CNNActor,self).__init__()
        self.conv1 = nn.Conv2d(4,32,4,2)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.fc1 = nn.Linear(18496, cf.hidden)
        self.pi = nn.Linear(cf.hidden,n_action)

        #### xavier init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self,x,dim=1):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1,18496)
        x = F.leaky_relu(self.fc1(x))
        prob = F.softmax(self.pi(x),dim = dim)
        return prob


class CNNCritic(nn.Module):
    def __init__(self):
        super(CNNCritic,self).__init__()
        self.conv1 = nn.Conv2d(4,32,4,2)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.fc1 = nn.Linear(18496, cf.hidden)
        self.fc_v = nn.Linear(cf.hidden,1)

        #### xavier init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self,x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1,18496)
        x = F.leaky_relu(self.fc1(x))
        v = self.fc_v(x)
        return v

###########
class NatureHead(nn.Module):
    def __init__(self):
        super(NatureHead, self).__init__()
        self.conv1 = nn.Conv2d(cf.stacked_frame,32,4,2)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.fc1 = nn.Linear(18496, cf.hidden)
        self.output_size = cf.hidden

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        ret = x.view(-1,18496)
        return ret


class ICM(torch.nn.Module):
    def __init__(self, action_space, state_size=18496, cnn_head=True):
        super(ICM, self).__init__()
        if cnn_head:
            self.head = NatureHead()

        self.forward_model = nn.Sequential(
            nn.Linear(state_size + action_space, 256),
            nn.ReLU(),
            nn.Linear(256, state_size))
        self.inverse_model = nn.Sequential(
            nn.Linear(state_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
            nn.ReLU())

    def forward(self, state, next_state, action):
        if hasattr(self, 'head'):
            phi1 = self.head(state)
            phi2 = self.head(next_state)
        else:
            phi1 = state
            phi2 = next_state
        phi2_pred = self.forward_model(torch.cat([action, phi1], 1)) # action -> actor(s)
        action_pred = F.softmax(self.inverse_model(torch.cat([phi1, phi2], 1)), -1)
        return action_pred, phi2_pred, phi1, phi2

