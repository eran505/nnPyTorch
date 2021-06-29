# Instantiate the env

import gym
from Game.simulator import Game_Sim



from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import ACER,ACKTR,DQN

import numpy as np

import gym

from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset

from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
dataset = ExpertDataset(expert_path='expert_cartpole.npz',
                        traj_limitation=1, batch_size=128)

model = PPO2('MlpPolicy', 'CartPole-v1', verbose=1)
# Pretrain the PPO2 model
model.pretrain(dataset, n_epochs=1000)

# As an option, you can train the RL agent
# model.learn(int(1e5))

# Test the pre-trained model
env = model.get_env()
obs = env.reset()

reward_sum = 0.0
for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        env.render()
        if done:
                print(reward_sum)
                reward_sum = 0.0
                obs = env.reset()

env.close()



exit()
######################################################################3
env = Game_Sim("/home/eranhe/car_model/debug")


model = ACKTR.load("./model/acktr")
ctr = 0
for _ in range(100):
    obs = env.reset()
    r_arr=[]

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        r_arr.append(rewards)
        if rewards>0:
            ctr+=1
        if dones:
            break

print("ctr:\t",ctr)