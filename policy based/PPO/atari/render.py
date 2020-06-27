import time
from agent import *
from env import *
from model import *
from utils import *
import config as cf
import torch
import numpy as np
from atari_wrappers import *
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# env = env_make(cf.env, action=None, ismario=cf.mario)
# env = gym.make(cf.env)
# env = wrap_deepmind(env,clip_rewards=False)
# env = env_make(cf.env, action='COMPLEX_MOVEMENT', ismario=cf.mario)
env = make_atari(cf.env)

n_action = env.action_space.n
agent = PPOagent(n_action=n_action, gamma=cf.gamma, lam=cf.lam, ent_coef=cf.ent_coef, learning_rate=cf.lr,
                 epoch=cf.epoch, batch_size=cf.batch_size, T=cf.horizon, eps=cf.eps, device=device)

agent.actor.load_state_dict(torch.load('model/atari/SpaceInvaders-v0/actor0-6500.pth'))
agent.critic.load_state_dict(torch.load('model/atari/SpaceInvaders-v0/critic0-6500.pth'))

while True:
    s = env.reset()
    s = prepro(s)
    state = np.zeros((cf.stacked_frame, cf.height, cf.width))
    for i in range(cf.stacked_frame):
        state[i, ::] = s
    done = False
    score = 0.0
    while not done:
        env.render()
        a, prob = agent.get_action(state)
        s_prime, r, done, info = env.step(a)
        state_prime = np.zeros_like(state)
        state_prime[:cf.stacked_frame - 1] = state[1:]
        state_prime[cf.stacked_frame - 1, :, :] = prepro(s_prime)
        score += r
        done_mask = 0 if done else 1
        s = s_prime
        if done:
            break
        # time.sleep(0.05)
    print(score)
env.close()




while True:
    done = False
    score = 0.0
    env.reset()
    while not done:
        env.render()
        s_prime, r, done, info = env.step(env.action_space.sample())
        score += r
        if done:
            break
        # time.sleep(0.05)
    print(score)
env.close()

with open('SpaceInvadersNoFrameskip-v0_2.pickle', 'rb') as f:
    data = pickle.load(f)
with open('AtlantisNoFrameskip-v0_2.pickle', 'rb') as f:
    data = pickle.load(f)
ma_data = make_ma(data,10)
plt.plot(data)
plt.plot(ma_data)

plt.show()
