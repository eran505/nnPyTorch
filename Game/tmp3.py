# Instantiate the env
import argparse

from simulator import Game_Sim
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines import ACER, ACKTR, DQN, PPO2
from stable_baselines3 import  PPO
import os
import numpy as np
# from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from time import time
import torch as th
import torch
from os_util import walk_rec

def get_config_csv(p):
    res = walk_rec(p,[],"csv")
    d={}
    for item in res:
        name = str(item).split(os.sep)[-1][:-4]
        id_exp = int(name.split("_")[-1])
        if id_exp not in d:
            d[id_exp]=[item]
        else:
            d[id_exp].append(item)
    for entry in d.keys():
        assert len(d[entry])==2
    return d

def inset_file_env(list_file,dest="./env_files/"):
    res= walk_rec(dest,[],"")
    for item in res:
        os.system("rm {}".format(item))
    for item in list_file:
        os.system("cp {} {}".format(item,dest))
    res = walk_rec(dest, [], "")
    assert len(res)==len(list_file)


parser = argparse.ArgumentParser(description='Process some integers.')


parser.add_argument('--steps', type=int,default=200000,
                    help='A required integer positional argument')

parser.add_argument('--mode', type=str, choices=["cpu","gpu"],default="cpu",
                    help='device name')


parser.add_argument('--eval_freq', type=int,default=5000,
                    help='policy eval freq')


parser.add_argument('--train', type=bool,default=True,
                    help='Train the RL agent')



args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.mode=="cpu":
    device='cpu'

print('Using device:', device)
# Additional Info when using cuda
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

if device=="cpu" and args.mode=="gpu":
   os.system("nvidia-smi")
   print(torch.backends.cudnn.enabled)
   os.system("nvcc --version")
   print(torch.__version__)
   print(torch.version.cuda)
   print(torch.cuda.is_available())
   assert False



policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[128,128,128], vf=[128,128,128])])


get_all_config = get_config_csv("./conf/")

for ky,val in get_all_config.items():

    path_to_log = "./logs/{}/".format(ky)
    inset_file_env(val)

    env = Game_Sim("./env_files/",is_option=False, action_mode='all',is_scale="None",discrete_actions=True)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1.0, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=path_to_log, n_eval_episodes=len(env.path_indexes),
                             log_path=path_to_log, eval_freq=args.eval_freq, callback_on_new_best=callback_on_best,
                             deterministic=True, verbose=1, render=False)

    if args.train:
        start_time = time()
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,batch_size=128,verbose=1).learn(total_timesteps=args.steps,callback=eval_callback)
        end_time = time()

        print( "Leraning Time:", end_time -start_time)
        with open(path_to_log+"time.txt","+w") as f:
            f.write("train time:{}".format(end_time -start_time))





    '''####################################### EVAL ###################################################'''


    model = PPO.load(path_to_log+"best_model.zip")

    num_iter = 100
    l_reward = np.zeros(num_iter)
    l_len_ep = np.zeros(num_iter)
    for j in range(num_iter):
        res = evaluate_policy(model, env, n_eval_episodes=1, return_episode_rewards=True)
        l_reward[j] = res[0][0]
        l_len_ep[j] = res[1][0]


    l_reward[l_reward < 1] = 0
    print("------ Policy Eval ------\nEpisodes: {}".format(num_iter))
    print("Collision rate:", np.mean(l_reward))
    print("Mean episodes length:", np.mean(l_len_ep))
    print("-------------------------")


    with open(path_to_log + "eval.txt", "+w") as f:
        f.write("Episodes:{}".format(num_iter))
        f.write("Collision rate:{}".format(np.mean(l_reward)))
        f.write("Mean episodes length:{}".format(np.mean(l_len_ep)) )

