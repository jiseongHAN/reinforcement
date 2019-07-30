'''
action_sapce box(3,)
-> [(-1,1),(0,1),(0,1)]

observation dim -> (96,96,3)

'''

import gym
import config as cf
import cv2
import matplotlib.pyplot as plt

env = gym.make(cf.env)

s = env.reset()
done = False
while not done:
    s_prime, r, done, info =  env.step(env.action_space.sample())
env.close()


# TODO : transition 저장하는 함수 코드 및 트레인
