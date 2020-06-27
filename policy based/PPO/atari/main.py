import pickle
from agent import *
from utils import *
import config as cf
import torch
from atari_wrappers import *
### make transition or roll-out
def main():
    reward_sum_running_avg = None
    memory = []
    n_train = 0
    n_epi = 0
    score_lst = []
    while True:
        # s = env.reset()._force()
        # s = s.transpose(2,0,1)
        s = env.reset()
        s = prepro(s)
        s = normalization(s,5)
        state = np.zeros((cf.stacked_frame,cf.height,cf.width))
        for i in range(cf.stacked_frame):
            state[i,::] = s
        score = 0.0
        step = 0
        done = False
        # a, prob = agent.get_action(s)
        reward_history = []
        while not done:
            for T in range(cf.horizon):
                if cf.render:
                    env.render()
                a, prob = agent.get_action(state)
                # a, prob = agent.get_action(s)
                s_prime, r, done, info = env.step(a)
                state_prime = np.zeros_like(state)
                state_prime[:cf.stacked_frame-1] = state[1:]
                state_prime[cf.stacked_frame-1, :, :] = normalization(prepro(s_prime),5)
                # s_prime = s_prime._force().transpose(2,0,1)
                score += r
                reward_history.append(r)
                # r = np.sign(r)
                done_mask = 0 if done else 1
                # data = (s, a, r, done_mask, s_prime)
                data = (state,a,r,done_mask,state_prime)
                # print('state = {}'.format((state == state_prime).all()))
                # print('fixed = {}'.format((s == prepro(s_prime)).all()))
                memory.append(data)
                # s = s_prime
                state = state_prime
                step += 1
                if done:
                    break
            n_train += 1
            # entropy = agent.train_net(memory)
            entropy = agent.train_net(memory)
            # print('#{} Successfully Trained - Entropy : {:.4f}'.format(n_train,entropy))
            memory = []
            # print('max : {:d} / {:f} \n action : {:d}'.format(torch.argmax(prob), torch.max(prob),a))
            reward_sum = sum(reward_history[-step:])
            reward_sum_running_avg = 0.99 * reward_sum_running_avg + 0.01 * reward_sum if reward_sum_running_avg else reward_sum
            if n_train % 500 == 0 and n_train != 0:
                model_save(agent.actor, cf.actor_path, cf.actor_name + '-'+ str(n_train))
                model_save(agent.critic, cf.critic_path, cf.critic_name+ '-' +  str(n_train))
                print('#{} : Successfully save model'.format(n_train))
        print('#{} 총 점수 : {} 스텝 : {} 평균 점수 : {}'.format(n_epi,score,step,reward_sum_running_avg))
        score_lst.append(score)
        # entropy_lst.append(entropy)
        if n_epi % 200 == 0 and n_epi != 0:
            with open(cf.env+'_2.pickle', 'wb') as f:
                pickle.dump(score_lst, f, pickle.HIGHEST_PROTOCOL)
            # with open(cf.env+'_entropy.pickle', 'wb') as f:
            #     pickle.dump(entropy_lst, f, pickle.HIGHEST_PROTOCOL)
            # # ### load
            # with open('data.pickle', 'rb') as f:
            #     data = pickle.load(f)
        score = 0.0
        step = 0.0
        n_epi += 1
    env.close()
# actor_optimizer,critic_optimizer,icm_optimizer



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_atari(cf.env)

    # env = gym.make(cf.env)
    # env = wrap_deepmind(env)
    # env = env_make(cf.env, action='COMPLEX_MOVEMENT', ismario=cf.mario)

    n_action = env.action_space.n
    agent = PPOagent(n_action = n_action, gamma = cf.gamma, lam = cf.lam, ent_coef= cf.ent_coef, learning_rate=cf.lr,
                     epoch = cf.epoch, batch_size= cf.batch_size, T = cf.horizon, eps= cf.eps, device = device)
    # actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
    # critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
    # icm_optimizer = optim.Adam(icm.parameters(), lr=0.001)

    if cf.resume:
        try:
            agent.actor.load_state_dict(torch.load(cf.actor_path + '/' + cf.actor_name + '.pth'))
            agent.critic.load_state_dict(torch.load(cf.critic_path + '/' + cf.critic_name+ '.pth'))
        except:
            print('Loading Failed Start from Scratch ')
    else:
        print('Start from Scratch')


    print('device => {}'.format(device))
    print('env => {}'.format(cf.env))
    print('skip => {}'.format(cf.skip))


    ##### Multiprocessing #####



    main()
# 자잘한 버그 / agent와 main 간의 호환! - env 안정성(클리어)
# TODO : hyperparameter 조정
# TODO : a3c / argparse

