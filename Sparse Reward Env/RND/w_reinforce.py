import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torch.distributions import Categorical


class RNetwork(nn.Module):
    def __init__(self, n_input):
        super(RNetwork, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.target = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.target.apply(self.random_init)

    def random_init(self, m): # for target_network
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight)

    def forward(self,x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x)
        pred = self.pred(x)
        target = self.target(x)
        return pred, target





class mlp(nn.Module):
    def __init__(self,n_obs,n_action):
        super(mlp,self).__init__()
        self.layer1 = nn.Linear(n_obs,128)
        self.layer2 = nn.Linear(128,128)
        self.q = nn.Linear(128,n_action)

        self.seq = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU(),
            self.q
        )

        self.seq.apply(init_weights)

    def forward(self,x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x)
            return torch.softmax(self.seq(x), dim=0)
        else:
            return torch.softmax(self.seq(x), dim =0)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)



def get_int(pred, target):
    mse = nn.MSELoss()
    loss_r = mse(target.detach(),pred)
    return float(loss_r)

def cal_returns(r_lst, gamma):
    ret_lst = []
    for i,j in enumerate(reversed(r_lst)):
        if i == 0:
            ret_lst.append(j)
        else:
            ret_lst.append(ret_lst[i-1]+ gamma * j)
    ret_lst.reverse()
    return ret_lst


def train(log_prob_lst, re_lst, ri_lst, ent_lst, gamma, optimizer, net_lst, opt):
    ri = [(i - np.mean(ri_lst))/np.mean(i) for i in ri_lst]
    r = [(2* i) + j for i,j in zip(re_lst,ri)]
    t_ret = cal_returns(r,gamma)
    # t_ret = cal_returns(re_lst,gamma)

    loss = 0
    loss_r = 0
    for i in range(len(t_ret)):
        loss -= t_ret[i] * log_prob_lst[i] + 0.0001 * ent_lst[i]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mse = nn.MSELoss()
    for i in range(len(net_lst)):
        loss_r +=  mse(net_lst[i][1].detach(),net_lst[i][0])
    opt.zero_grad()
    loss_r.backward()
    opt.step()

def main():
    gamma = 0.98
    env = gym.make("CartPole-v1")
    epoch = 10000
    q = mlp(env.observation_space.shape[0],env.action_space.n)
    net = RNetwork(env.observation_space.shape[0])
    opt = optim.Adam(net.pred.parameters(),lr=0.00025)
    optimizer = optim.Adam(q.parameters(),lr=0.00025)
    see = []
    for i in range(epoch):
        ri_lst = []
        re_lst = []
        log_prob_lst = []
        s = env.reset()
        done = False
        total_score = 0
        ext_score = 0
        net_lst = []
        ent_lst = []
        while not done:
            # env.render()
            m = Categorical(q(s))
            a = m.sample()
            s_prime, ri, done, _ = env.step(int(a))
            pred, target = net(s_prime)
            re = get_int(pred,target)
            net_lst.append((pred,target))
            s = s_prime
            total_score += ri
            ext_score += re
            ent_lst.append(m.entropy())
            re_lst.append(np.clip(re,-1,1))
            ri_lst.append(ri)
            log_prob_lst.append(m.log_prob(a))

        train(log_prob_lst, re_lst, ri_lst, ent_lst ,gamma, optimizer,net_lst,opt)
        print("Epoch : %d | score : %f | r_i : %f" %(i, total_score,ext_score))
        see.append(total_score)




if __name__ ==  "__main__":
    main()


