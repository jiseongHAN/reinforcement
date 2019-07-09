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
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = x.view(-1,18496)
        x = swish(self.fc1(x))
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
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = x.view(-1,18496)
        x = swish(self.fc1(x))
        v = self.fc_v(x)
        return v


class ICMModel(nn.Module):
    def __init__(self,n_action):
        super(ICMModel,self).__init__()
        feature_output = 18496
        self.conv1 = nn.Conv2d(4,32,4,2)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.fc1 = nn.Linear(18496, cf.hidden)
        self.inv = nn.Linear(cf.hidden*2, cf.hidden)
        self.fc_pi = nn.Linear(cf.hidden,n_action)
        self.res = nn.Linear(n_action + cf.hidden,cf.hidden)
        self.res1 = nn.Linear(cf.hidden,cf.hidden)
        self.n_action = n_action

        #### xavier init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.inv.weight)
        nn.init.xavier_uniform_(self.res.weight)


    def feature(self,x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = x.view(-1,18496)
        x = self.fc1(x)
        return x
    def inverse(self,x):
        x = swish(self.inv(x))
        x = F.softmax(self.fc_pi(x),1)
        return x
    def residual(self,x):
        x = swish(self.res(x))
        x = self.res1(x)
        x = [x]
        x = x * 8
        return x
    def forward_net1(self,x):
        x = swish(self.res(x))
        return x

    def forward_net2(self,x):
        return self.res(x)

    def forward(self,inputs):
        state, action, s_prime= inputs
        action = torch.LongTensor(action)
        action_onehot = torch.FloatTensor(
            len(action), self.n_action)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)
        encode_state = self.feature(state)
        encode_next_state = self.feature(s_prime)

        ### get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse(pred_action)
        ######################################
        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action_onehot), 1)
        pred_next_state_feature_orig = self.forward_net1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual(torch.cat((pred_next_state_feature_orig, action_onehot), 1))[i * 2]
            pred_next_state_feature_orig = self.residual(
                torch.cat((pred_next_state_feature, action_onehot), 1))[i * 2 + 1] + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net2(torch.cat((pred_next_state_feature_orig, action_onehot), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action

