from agent import *
from env import *
from model import *
from utils import *
import config as cf
import torch
import numpy as np
from torch.distributions.categorical import Categorical
import torch.optim as optim

### make transition or roll-out
def main():
    env = env_make(cf.env,ismario=cf.mario)
    memory = []
    n_train = 0
    n_epi = 0
    while True:
        s = env.reset()
        s = prepro(s)
        state = np.zeros((cf.stacked_frame,cf.height,cf.width))
        for i in range(cf.stacked_frame):
            state[i,::] = s
        score = 0.0
        step = 0
        done = False
        while not done:
            for T in range(cf.horizon):
                if cf.render:
                    env.render()
                prob = actor(torch.tensor(state,dtype=torch.float).unsqueeze(0).to(device))
                action = Categorical(prob.cpu())
                a = action.sample().item()
                s_prime, r, done, info = env.step(a)
                state_prime = np.zeros_like(state)
                state_prime[:cf.stacked_frame-1] = state[1:]
                state_prime[cf.stacked_frame-1, :, :] = prepro(s_prime)
                score += r
                done_mask = 0 if done else 1
                data = (state,a,r,state_prime,done_mask, prob.squeeze()[a].item())
                # print('state = {}'.format((state == state_prime).all()))
                # print('fixed = {}'.format((s == prepro(s_prime)).all()))
                memory.append(data)
                state = state_prime
                step += 1
                if done:
                    break
            n_train += 1
            train_net(memory,actor,critic,icm,optimizer)
            print('#{} Successfully Trained'.format(n_train))
            memory = []
            if n_train % 500 == 0 and n_train != 0:
                model_save(actor, cf.actor_path, cf.actor_name)
                model_save(critic, cf.critic_path, cf.critic_name)
                model_save(icm, cf.icm_path, cf.icm_name)
                print('#{} : Successfully save model'.format(n_train))
        print('#{} 총 점수 : {} 스텝 : {}'.format(n_epi,score,step))
        score = 0.0
        step = 0.0
        n_epi += 1
    env.close()
# actor_optimizer,critic_optimizer,icm_optimizer



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device => {}'.format(device))

    n_action = gym.make(cf.env).action_space.n
    critic = nn.DataParallel(CNNCritic()).to(device)
    actor = nn.DataParallel(CNNActor(n_action)).to(device)
    icm = nn.DataParallel(ICM(n_action)).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.0002)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.0002)
    icm_optimizer = optim.Adam(icm.parameters(), lr=0.0002)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()) + list(icm.parameters()), lr=cf.lr,
                           weight_decay=cf.l2_rate)  # weight_decay

    if cf.resume:
        try:
            actor.load_state_dict(torch.load(cf.actor_path + '/' + cf.actor_name))
            critic.load_state_dict(torch.load(cf.critic_path + '/' + cf.critic_name))
            icm.load_state_dict(torch.load(cf.icm_path + '/' + cf.icm_name))
        except:
            print('Loading Failed Start from Scratch ')
    else:
        print('Start from Scratch')

    main()
# 자잘한 버그 / agent와 main 간의 호환! - env 안정성(클리어)
# TODO : hyperparameter 조정
# TODO : GPU 버전 만들기(O) / + a3c(O) / argparse
