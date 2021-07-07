import numpy as np
import random

class BernoulliBandit:
    def __init__(self):
        self.seed = np.random.random()
    def get_reward(self):
        if np.random.random() < self.seed:
            return 1
        else:
            return 0

def rargmax(vector):
    """
    Random argmax function.
    Reference: https://gist.github.com/dalek7/37b108d42c3ec5b3e90049a72f828e22
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)


mach = [BernoulliBandit() for _ in range(10)]

#Random Policy
total_reward = 0
for _ in range(100):
    each_reward = [0] * len(mach)
    reward_lst = []
    _selected = [1e-7] * len(mach)
    for _ in range(1000):
        action = np.random.randint(0,len(mach))
        reward = mach[action].get_reward()
        total_reward += reward
        each_reward[action] += reward
        _selected[action] += 1
        reward_lst.append(total_reward)

print("total_reward: ", total_reward / 100)
print("_selected: ", _selected)


#Greedy Policy
total_greed = 0
for _ in range(100):
    each_greed = [0] * len(mach)
    greed_lst = []
    greed_selected = [1e-7] * len(mach)

    for _ in range(1000):
        means = [ucb / actions for ucb, actions in zip(each_greed, greed_selected)]
        action = rargmax(means)
        reward = mach[action].get_reward()
        total_greed += reward
        each_greed[action] += reward
        greed_lst.append(total_greed)
        greed_selected[action] += 1
print("total greed: ", total_greed / 100)
print("greed_selected: ", greed_selected)

#e-greedy Policy
total_egreed = 0
for _ in range(100):
    each_egreed = [0] * len(mach)
    egreed_lst = []
    egreed_selected = [1e-7] * len(mach)

    for _ in range(1000):
        if np.random.normal() < 0.01:
            action = np.random.randint(0,len(mach))
        else:
            means = [ucb / actions for ucb, actions in zip(each_egreed, egreed_selected)]
            action = rargmax(means)
        reward = mach[action].get_reward()
        total_egreed += reward
        each_egreed[action] += reward
        egreed_selected[action] += 1
        egreed_lst.append(total_egreed)
print("total egreed: ", total_egreed / 100)
print("egreed_selected: ", egreed_selected)



#UCB1 Policy
total_ucb = 0

for _ in range(100):
    each_ucb = [0] * len(mach)
    action_selected = [1e-7] * len(mach)
    ucb_lst = []
    for t in range(1,1001):
        means = [ucb / actions for ucb, actions in zip(each_ucb, action_selected)]
        ucb = [np.sqrt(2 * np.log(t) / selected) for selected in action_selected]
        ucb = np.array(means) + np.array(ucb)
        action = rargmax(ucb)
        action_selected[action] += 1
        reward = mach[action].get_reward()
        total_ucb += reward
        each_ucb[action] += reward
        ucb_lst.append(total_ucb)
print("total_ucb: ", total_ucb / 100)
print("action_selected: ", action_selected)