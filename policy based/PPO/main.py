import argparse
import torch.optim as optim
import torch
from model import *
from train_ppo import *
import gym
from collections import deque
import hp
import numpy as np
import math
# env = gym.make('MountainCarContinuous-v0')


def main():
    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]
    input_dim = env.observation_space.shape[0]
    actor = Actor(input_dim, action_dim)
    buffer = []
    critic = Critic(input_dim)
    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                              weight_decay=hp.l2_rate)

    if hp.resume:
        try:
            actor.load_state_dict(torch.load(hp.actor_path + '/' + hp.actor_name))
            critic.load_state_dict(torch.load(hp.critic_path + '/' + hp.critic_name))
        except:
            print('Loading Failed Start from Scratch ')
    else:
        print('Start from Scratch')
    n_train = 0
    loss_lst = []
    for _ in range(hp.iter_max):
        s = env.reset()
        done = False
        score = 0.0
        step =0
        while not done:
            for _ in range(hp.rollout_len):
                # env.render()
                # s = norm_state(s)
                mu, std, logstd = actor(torch.from_numpy(s).float())
                a = get_action(mu,std)
                s_prime, r, done, _ = env.step(torch.clamp(torch.Tensor(a),-1,1).data.numpy())
                score += r
                r = np.clip(r,-5,5)
                done_mask = 0 if done else 1
                # s_prime = norm_state(s_prime)
                sarsd = (s,a,r,s_prime, done_mask)
                buffer.append(sarsd)
                step += 1
                s = s_prime
                if done:
                    break
            loss = train(actor,critic,buffer,actor_optim,critic_optim)
            loss_lst.append(loss)
            n_train += 1
            buffer = []
            if n_train % 100 == 0 and n_train != 0:
                model_save(actor, hp.actor_path, hp.actor_name)
                model_save(critic, hp.critic_path, hp.critic_name)
                print('#{} : Successfully save model'.format(n_train))

        if score > 200.0:
            print('Clear!!')
            break
        print('#{} : 총 점수 : {} 총 스텝 : {}'.format(n_train,score,step))
        score = 0.0
        step = 0
    env.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO for Continuous Action Space - support Box env')
    parser.add_argument('--env', default='LunarLanderContinuous-v2', help='Name of environment')
    parser.add_argument('--resume', default=hp.resume, help='Resume or Start from Scratch')
    parser.add_argument('--actor', default=hp.actor_path, help='Path for saving actor-model')
    parser.add_argument('--critic', default=hp.critic_path, help = 'Path for saving critic-model')

    args = parser.parse_args()
    env_name = args.env
    hp.resume = args.resume
    hp.actor_path = args.actor
    hp.critic_path = args.critic


    main()

# TODO : use argparse to run py
