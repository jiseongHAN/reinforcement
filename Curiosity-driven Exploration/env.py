import cv2
import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from collections import deque

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


# TODO : env에서 액션부분 떼어 버리기 + stack 함수 만들기 / 하나로 합치는 부분 고민
class MarioEnv():
    def __init__(self,env_id:str,stack_frame:int,h:int,w:int,skip:int):
        self.h = h
        self.w = w
        self.stack_frame = stack_frame
        self.env = gym_super_mario_bros.make(env_id)
        self.env = JoypadSpace(self.env,COMPLEX_MOVEMENT)
        self.env = SkippedEnv(self.env,skip)
        self.dummy = np.zeros((self.stack_frame,self.h,self.w))
        self.box = deque(maxlen=stack_frame)

    def prepro(self,state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state,(self.h,self.w)) / 255.0
        return state

    def get_init_state(self,state):
        for i in range(self.stack_frame):
            self.box.append(self.prepro(state))
        return self.box

    def reset(self): # reset env -> return (f,w,h)
        obs = self.get_init_state(self.env.reset())
        return obs

    def run(self,actions): # s, a, r, s_prime, done_mask
        s = self.reset()
        done = False
        while not done:
            s_prime,r,done,info = self.env.step(actions)
            s_prime = self.prepro(s_prime)
            box = s.append(s_prime)
            done_mask = 0 if done else 1
            data = (s,actions,r,box,done_mask)
            s = box
        pass



