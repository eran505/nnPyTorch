# Instantiate the env

from Game.simulator import Game_Sim
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines import ACER, ACKTR, DQN, PPO2
from stable_baselines3 import  PPO

import numpy as np
# from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from time import time
import torch as th
'''################################   Custom Networks   ######################################'''
# class CustomPolicy(FeedForwardPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs,
#                                                net_arch=[128,dict(pi=[128, 128, 128],
#                                                           vf=[128, 128, 128])],
#                                            feature_extraction="mlp")
#
# register_policy('CustomPolicy', CustomPolicy)

'''#################################### Train ######################################################'''


policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[128, 128,128], vf=[128, 128,128])])

env = Game_Sim("/home/eranhe/car_model/debug",False)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1.0, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/', n_eval_episodes=100,
                             log_path='./logs/', eval_freq=1000, callback_on_new_best=callback_on_best,
                             deterministic=True, verbose=1,     render=False)
start_time = time()
# model = PPO2('CustomPolicy', env).learn(total_timesteps=200000,callback=eval_callback)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1).learn(total_timesteps=200000,callback=eval_callback)


end_time = time()
print( "Leraning Time:", end_time -start_time)
with open("./logs/time.txt","+w") as f:
    f.write("train time:{}".format(end_time -start_time))

# MlpPolicy


'''####################################### EVAL ###################################################'''


model = PPO.load("./logs/best_model.zip")

num_iter = 100
l_reward = np.zeros(num_iter)
l_len_ep = np.zeros(num_iter)
for j in range(num_iter):
    res = evaluate_policy(model, env, n_eval_episodes=1, return_episode_rewards=True)
    l_reward[j] = res[0][0]
    l_len_ep[j] = res[1][0]

l_reward[l_reward < 0] = 0
print("------ Policy Eval ------\nEpisodes: {}".format(num_iter))
print("Collision rate:", np.mean(l_reward))
print("Mean episodes length:", np.mean(l_len_ep))
print("-------------------------")
