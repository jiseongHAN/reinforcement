import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
import numpy as np
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

class FlagReward(gym.Wrapper):
    def __init__(self,env):
        gym.Wrapper.__init__(self,env)
        self.score = 0.0
    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        reward += (info['score'] - self.score) / 100.
        self.score = info['score']
        # if info['life'] <= 1:
        #     done = True
        if done:
            if info['flag_get']:
                reward += 15
            else:
                reward -= 15
        return obs, reward, done, info

    def reset(self):
        self.score = 0.0
        return self.env.reset()

def env_make(env_id, action):
    mario = gym_super_mario_bros.make(env_id)
    env = FlagReward(mario)
    env = SkippedEnv(env, cf.skip)
    if action == COMPLEX_MOVEMENT:
        env = JoypadSpace(env,COMPLEX_MOVEMENT)
    elif action == SIMPLE_MOVEMENT:
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
    elif action == RIGHT_ONLY:
        env = JoypadSpace(env,RIGHT_ONLY)
    else:
        raise NotImplementedError
    return env

