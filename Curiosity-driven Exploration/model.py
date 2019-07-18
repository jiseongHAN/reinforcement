import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cf

def swish(x):
    ret = x * torch.sigmoid(x)
    return ret

class CNNActor(nn.Module):
    def __init__(self,n_action):
        super(CNNActor,self).__init__()
        self.conv1 = nn.Conv2d(cf.stacked_frame,32,4,2)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,2,1)
        self.fc1 = nn.Linear(20736, cf.hidden)
        self.pi = nn.Linear(cf.hidden,n_action)

        #### xavier init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self,x,dim=1):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = x.view(-1,20736)
        x = swish(self.fc1(x))
        prob = F.softmax(self.pi(x),dim = dim)
        return prob


class CNNCritic(nn.Module):
    def __init__(self):
        super(CNNCritic,self).__init__()
        self.conv1 = nn.Conv2d(cf.stacked_frame,32,4,2)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,2,1)
        self.fc1 = nn.Linear(20736, cf.hidden)
        self.fc_v = nn.Linear(cf.hidden,1)

        #### xavier init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self,x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = x.view(-1,20736)
        x = swish(self.fc1(x))
        v = self.fc_v(x)
        return v

###########
class NatureHead(nn.Module):
    def __init__(self):
        super(NatureHead, self).__init__()
        self.conv1 = nn.Conv2d(cf.stacked_frame,32,4,2)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,2,1)
        self.fc1 = nn.Linear(20736, cf.hidden)
        self.output_size = cf.hidden

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        ret = x.view(-1,20736)
        return ret


class ICM(torch.nn.Module):
    def __init__(self, action_space, state_size=20736, cnn_head=True):
        super(ICM, self).__init__()
        if cnn_head:
            self.head = NatureHead()

        self.forward_model = nn.Sequential(
            nn.Linear(state_size + action_space, cf.hidden),
            nn.ReLU(),
            nn.Linear(cf.hidden, state_size))
        self.inverse_model = nn.Sequential(
            nn.Linear(state_size * 2, cf.hidden),
            nn.ReLU(),
            nn.Linear(cf.hidden, action_space),
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

