import cv2
import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from collections import deque
import config as cf

class SkippedEnv(gym.Wrapper):
    def __init__(self,env,skip):
        gym.Wrapper.__init__(self,env)
        self.n_skip = skip

    def step(self,action):
        total_reward = 0.0
        done = None
        obs_lst =[] #TODO : 메모리 효율 체크
        for i in range(self.n_skip):
            obs, reward, done, info = self.env.step(action)
            obs_lst.append(obs)
            total_reward += reward
            if done:
                break
        max_obs = np.stack(obs_lst).max(axis = 0)
        return max_obs, reward, done, info
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# TODO : env.render() 추가
# TODO : env에서 액션부분 떼어 버리기 + stack 함수 만들기 / 하나로 합치는 부분 고민

def prepro(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state,(cf.height,cf.width)) / 255.0
    return state
