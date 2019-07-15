import torch.multiprocessing as mp
from agent import *
from env import *
from model import *
from utils import *
import config as cf
import torch
import random
import numpy as np
from torch.distributions.categorical import Categorical
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from tensorboardX import SummaryWriter


### make transition or roll-out
def main(env_id,action_space,actor,critic,icm,optimizer, save,index=0):
    env = env_make(env_id, action_space)
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
                state_prime[:3] = state[1:]
                state_prime[3, :, :] = prepro(s_prime)
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
                if len(memory) >= 3200:
                    n_train += 1
                    train_net(memory,actor,critic,icm,optimizer)
                    print('Worker{} - #{} Successfully Trained'.format(index+1,n_train))
                    memory = []
                    if save and n_train % 100 == 0 and n_train != 0:
                        model_save(actor, cf.actor_path, cf.actor_name)
                        model_save(critic, cf.critic_path, cf.critic_name)
                        model_save(icm, cf.icm_path, cf.icm_name)
                        print('Worker{} - #{} : Successfully save model'.format(index+1,n_train))
            print('Worker{} - #{} 총 점수 : {} 스텝 : {} 스테이지 : {}'.format(index+1, n_epi,score,step,info['stage']))
        score = 0.0
        step = 0.0
        n_epi += 1
    env.close()
# actor_optimizer,critic_optimizer,icm_optimizer



if __name__ == '__main__':
    if cf.gpu and cf.a3c:
        raise NotImplementedError
    if cf.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device => {}'.format(device))

    else:
        device = 'cpu'
        print('device => {}'.format(device))


    torch.manual_seed(123)
    mp.set_start_method('spawn')


    if cf.action_space == 0:
        n_action = 12
        action_space = COMPLEX_MOVEMENT
    elif cf.action_space == 1:
        n_action = 7
        action_space = SIMPLE_MOVEMENT
    else:
        n_action = 5
        action_space = RIGHT_ONLY


    critic = CNNCritic().to(device)
    actor = CNNActor(n_action).to(device)
    icm = ICM(n_action).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.0002)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.0002)
    icm_optimizer = optim.Adam(icm.parameters(), lr=0.0002)


    if cf.resume:
        try:
            actor.load_state_dict(torch.load(cf.actor_path + '/' + cf.actor_name))
            critic.load_state_dict(torch.load(cf.critic_path + '/' + cf.critic_name))
            icm.load_state_dict(torch.load(cf.icm_path + '/' + cf.icm_name))
        except:
            print('Loading Failed Start from Scratch ')
    else:
        print('Start from Scratch')

    if cf.a3c:
        critic.share_memory()
        actor.share_memory()
        icm.share_memory()
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()) + list(icm.parameters()), lr=cf.lr,
                           weight_decay=cf.l2_rate)  # weight_decay

    if cf.a3c:
        processes = []
        for index in range(cf.num_processes):
            if index == 0:
                process = mp.Process(target=main, args=('SuperMarioBros-1-{}-v0'.format(index+1),action_space,actor,critic,icm,optimizer, True, index))
            else:
                process = mp.Process(target=main,args = ('SuperMarioBros-1-{}-v0'.format(index+1),action_space,actor,critic,icm,optimizer, False, index))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

        print('START!!!!!!')
    else:
        main(cf.env,action_space,actor,critic,icm,optimizer, False)

# 자잘한 버그 / agent와 main 간의 호환! - env 안정성(클리어)
# TODO : hyperparameter 조정
# TODO : GPU 버전 만들기(O) / + a3c / argparse
