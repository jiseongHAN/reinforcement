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
        self.conv1 = nn.Conv2d(1,32,4,2)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,4,2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,2,1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(28224, cf.hidden)
        self.pi = nn.Linear(cf.hidden,n_action)

        #### xavier init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self,x,dim=1):
        x = torch.relu(self.conv1(x))
        # x = self.bn1(x)
        x = torch.relu(self.conv2(x))
        # x = self.bn2(x)
        x = torch.relu(self.conv3(x))
        # x = self.bn3(x)
        x = x.view(-1,28224)
        x = torch.relu(self.fc1(x))
        mu = self.pi(x)
        mu[0] = torch.tanh(mu[0])
        mu[1:3] = torch.sigmoid(mu[1:3])
        return mu


class CNNCritic(nn.Module):
    def __init__(self):
        super(CNNCritic,self).__init__()
        self.conv1 = nn.Conv2d(1,32,4,2)
        # self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,4,2)
        # self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,64,2,1)
        # self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(28224, cf.hidden//2)
        self.fc_a = nn.Linear(3,cf.hidden//2)
        self.fc_v = nn.Linear(cf.hidden,1)

        #### xavier init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self,x,a):
        x = torch.relu(self.conv1(x))
        # x = self.bn1(x)
        x = torch.relu(self.conv2(x))
        # x = self.bn2(x)
        x = torch.relu(self.conv3(x))
        # x = self.bn3(x)
        x = x.view(-1,28224)
        x = torch.relu(self.fc1(x))
        a = torch.relu(self.fc_a(a))
        v = torch.cat([x,a], dim = 1)
        v = self.fc_v(v)
        return v
###########



# TODO : 모델 만들기 Q / Value
