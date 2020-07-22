import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torch.distributions import Categorical


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

def cal_returns(r_lst, gamma):
    ret_lst = []
    for i,j in enumerate(reversed(r_lst)):
        if i == 0:
            ret_lst.append(j)
        else:
            ret_lst.append(ret_lst[i-1]+ gamma * j)
    ret_lst.reverse()
    return ret_lst


def train(log_prob_lst, reward_lst, ent_lst, gamma, optimizer):
    t_ret = cal_returns(reward_lst,gamma)
    loss = 0
    loss_r = 0
    for i in range(len(t_ret)):
        loss -= t_ret[i] * log_prob_lst[i] + 0.0001 * ent_lst[i]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    gamma = 0.98
    env = gym.make("CartPole-v1")
    epoch = 10000
    q = mlp(env.observation_space.shape[0],env.action_space.n)
    optimizer = optim.Adam(q.parameters(),lr=0.00025)
    for i in range(epoch):
        reward_lst = []
        log_prob_lst = []
        s = env.reset()
        done = False
        total_score = 0
        ent_lst = []
        while not done:
            # env.render()
            m = Categorical(q(s))
            a = m.sample()
            s_prime, r, done, _ = env.step(int(a))
            s = s_prime
            total_score += r
            ent_lst.append(m.entropy())
            reward_lst.append(np.clip(r,-1,1))
            log_prob_lst.append(m.log_prob(a))

        train(log_prob_lst, reward_lst, ent_lst ,gamma, optimizer)
        print("Epoch : %d | score : %f " %(i, total_score))




if __name__ ==  "__main__":
    main()


