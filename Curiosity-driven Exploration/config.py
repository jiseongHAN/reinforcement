env = 'SuperMarioBros-v0'
height = 84
width = 84
stacked_frame = 4
skip = 6
gamma = 0.95
lam = 0.98
eps = 0.2
lr = 0.0002
epoch = 5
batch_size = 64
hidden = 512
iter_max = 10000
l2_rate = 0.002
resume = False
actor_path = 'model/complex'
critic_path = 'model/complex'
icm_path = 'model/complex'
actor_name = 'actor_a3c'
critic_name = 'critic_a3c'
icm_name = 'icm_a3c'
render = False
horizon = 1024
num_processes = 4
action_space = 0
a3c = False
gpu = True
