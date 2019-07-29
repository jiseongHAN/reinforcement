'''
inverse loss = cross entropy ( pred_action / real_action)
forward loss = mse(pred_next_feature, real_next_feature)
actor loss = -torch.min(surr,1-eps,1+eps)
critic loss = mse(v, td_target)
total loss = actor + 0.5 * critic + forward + inverse
there is a Beta in the paper but I don't know why there is no beta in implementation code

Agent has to calculate td_target, adv, pred_action, pred_feature from (s,a,r,s_prime,done_mask)
'''
# TODO : tensorboard로 visualization
from model import *
from utils import *
import config as cf
import torch
import random
from torch.distributions import Categorical
from collections import deque
import torch.optim as optim
import torch.nn.functional as F
### agent



# TODO : agent class로 합치기?
#### make batch from memory
# actor_optimizer, critic_optimizer, icm_optimizer
class PPOagent():
    def __init__(self,
                 n_action,
                 gamma,
                 lam,
                 ent_coef,
                 learning_rate,
                 epoch,
                 batch_size,
                 T,
                 eps,
                 device):
        self.n_action = n_action
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.T = T
        self.eps = eps
        self.device = device
        self.actor = CNNActor(self.n_action)
        self.critic = CNNCritic()
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()))



    def get_action(self,state):
        state = torch.tensor(state, dtype = torch.float)
        state = state.unsqueeze(0).to(self.device)
        prob = self.actor(state)
        action = Categorical(prob.cpu())
        # action = Categorical(F.softmax(prob, dim=-1).data.cpu().numpy())
        a = action.sample().item()
        return a , prob

    def train_net(self,memory):
        mse = nn.MSELoss()
        entropy_lst = deque(maxlen=cf.batch_size)
        idx = np.arange(len(memory))
        s_lst, a_lst, r_lst, done_lst, s_prime_lst = [], [], [], [], []
        for transition in memory:
            s, a, r, done, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
        s = torch.tensor(s_lst, dtype=torch.float).to(self.device)
        a = torch.tensor(a_lst).to(self.device)
        r = torch.tensor(r_lst, dtype=torch.float).to(self.device)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float).to(self.device)
        done = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        action_probs = self.actor(s)
        old = action_probs.gather(1,a).detach()  ################################################### DETACH 유무의 차이
        # m_old = Categorical(action_probs)
        # log_old = m_old.log_prob(a.squeeze())
        # inverse_loss, forward_loss, intrinsic_reward = get_icm_loss(s, s_prime, a, action_probs,icm)
        # total_reward = r + intrinsic_reward.unsqueeze(-1)
        total_reward = r
        # pred_action = pred_action.gather(1,a)
        ########### advantage 구하기!!!!!!!!!!!!!!!!!!! ##############
        old_v = self.critic(s).detach()
        td_target = total_reward + cf.gamma * self.critic(s_prime) * done
        td_target = td_target.detach()
        delta = td_target - old_v
        #
        advantage_lst = []
        advantage = 0.0
        for t in reversed(range(len(r))):
            advantage = self.gamma * self.lam * advantage * done[t][0] + delta[t][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        adv= torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        #
        adv = normalization(adv)
        # adv = normalization(td_target)


        for _ in range(cf.epoch):
            random.shuffle(idx)
            for n in range(len(idx) // self.batch_size):
                index = idx[n * self.batch_size : (n+1) * self.batch_size]
                new_v = self.critic(s[index])
                pi = self.actor(s[index])
                m = Categorical(pi)
                # log_pi = m.log_prob(a[index].squeeze())
                new_pi = pi.gather(1,a[index])
                # pi_a = pi.gather(1,a)
                ratio = new_pi / old[index]
                surr = ratio * adv[index]
                clip = torch.clamp(surr, 1 - self.eps, 1 + self.eps) * adv[index]
                actor_loss = -torch.min(clip,surr).mean()
                critic_loss = mse(td_target[index], new_v)

                entropy = m.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy
                # loss = actor_loss + 0.5 * critic_loss

                # loss = actor_loss - cf.ent_coef * entropy

                # loss = actor_loss + 0.5 * critic_loss+ inverse_loss + forward_loss
                # actor_optimizer.zero_grad()
                # loss.backward(retain_graph = True)
                # actor_optimizer.step()
                #
                # critic_optimizer.zero_grad()
                # loss.backward(retain_graph = True)
                # critic_optimizer.step()
                # #
                # icm_optimizer.zero_grad()
                # loss.backward(retain_graph = True)
                # icm_optimizer.step()
                # torch.cuda.empty_cache()

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                # print(loss)
                entropy_lst.append(entropy.cpu().detach().data.numpy())


        return sum(entropy_lst)/(len(entropy_lst) + 1e-10)


