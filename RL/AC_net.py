from abc import ABC

from sklearn.kernel_approximation import RBFSampler
import sklearn.preprocessing
import numpy as np
import gym
from os import getcwd
from os import path
from time import time
from sklearn import preprocessing
from sklearn import pipeline
from torch import nn
import torch
import random
import pandas as pd
import os
import torch.nn.functional as F
from PRB.replay_buffer import ReplayBuffer,PrioritizedReplayBuffer

#random.seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

cwd = getcwd()

def load_buffer(p="/home/eranhe/car_model/debug/data.csv"):
    size = 2500000
    my_replay_buffer = ReplayBuffer(size)
    data = pd.read_csv(p)
    print(len(data))
    data = data.to_numpy()
    for i, item in enumerate(data):
        my_replay_buffer.push(item[:12], item[12] * 26, item[25], item[13:25], item[26])
        i = i + 1
        if i > size:
            print("[replay_buffer] not all DATA")
            break
    return my_replay_buffer


def eval_test(test_buffer, policy):
    correct = 0
    iter_number = 40
    for _ in range(iter_number):
        state, action, reward, next_state, done = test_buffer.sample(1)
        actionz_pred = policy.select_action(state)
        correct += (actionz_pred == int(action[0]))
    print("test - correct:\t", correct / float(iter_number))
    return correct / float(iter_number)


class ActorNet(nn.Module, ABC):
    def __init__(self, input_dims, fc1_dim, fc2_dim, action_num, alpha):
        super(ActorNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dim
        self.fc2_dims = fc2_dim
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, action_num)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
        self.epslion = 0.05
        self.log_prob = None
        self.checkpoint_file = path.join(cwd, "modelz/actorNet")

    def forward(self, observation):
        state = torch.squeeze(observation)
        x = F.relu_(self.fc1(state))
        x = F.relu_(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=0)
        return x

    def sample_action(self, actions_probabilities):
        action = actions_probabilities.sample()
        return action.item()

    def sample_policy(self, state,action):
        pi = self.forward(state)
        probabilities = torch.distributions.Categorical(pi)
        log_probs = probabilities.log_prob(action)
        return log_probs,probabilities

    def save(self,suffix=""):
        torch.save(self.state_dict(), self.checkpoint_file+str(suffix)+".pt")

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object):
    def __init__(self,critic_file):
        self.gamma = 1.0
        self.log_prob = None
        self.features_number = 12
        self.actor = None
        self.critic = None
        self.critic_target = None
        self.learn_at = 100
        self.memo = None
        self.ctr = 0
        self.update_target_estimator_every = 1000
        self.init_networks()
        self.df_critic= pd.read_csv(critic_file)
    def get_learning_steps(self):
        return self.ctr

    def init_networks(self):
        self.actor = ActorNet(input_dims=self.features_number, fc1_dim=256, fc2_dim=256, action_num=27, alpha=0.01)

    def get_real_q_val(self,np_state,df,action):
        #print(np_state)
        new_df = df[((df[str(0)] == np_state[0]) & (df[str(2)] == np_state[2]) & (df[str(10)] == np_state[10]) &  (df[str(5)] == np_state[5]) )]
        if len(new_df)==0:
            return None
        return new_df[str(int(action+12))].values[-1]

    def learn(self,buffer):
        self.ctr += 1

        state, action, reward, new_state, done = buffer.sample(1)
        reward = torch.tensor(reward).float().to(device)
        done = torch.tensor(done).float().to(device)
        state_next = torch.tensor(new_state).float().to(device)
        state_tensor = torch.tensor(state).float().to(device)
        action_tensor = torch.tensor(action,dtype=torch.int64)
        state = np.squeeze(state)

        value = self.get_real_q_val(state,self.df_critic,action[0])
        #value_next =  self.get_target_val(state_next)
        #value_next = self.critic(state_next)


        #advantage = reward + (1.0 - done) * self.gamma * value_next - value
        advantage = value
        if advantage is None:
            print("miss")
            return

        log_probs,probabilities = self.actor.sample_policy(state_tensor,action_tensor)

        a = self.actor.sample_action(probabilities)

        print("net:{} real:{}".format(int(a),int(action[0])))

        actor_loss = -log_probs #* advantage

        #print(actor_loss)
        self.actor.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()






    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        f_observation = self.Rbf.featurize_state(observation)
        f_observation = torch.from_numpy(f_observation).float().to(device)
        actions ,log_prob = self.actor.sample_policy(f_observation)
        self.log_prob=log_prob
        return actions.item()

    def save_buffer(self, observation, action, reward, observation_, done):
        self.memo = [self.Rbf.featurize_state(observation), action, reward, self.Rbf.featurize_state(observation_),
                     done]

    def save_nets(self,index_time):
        self.actor.save(index_time)
        self.critic.save(index_time)

    def get_target_val(self, state_s):
        with torch.no_grad():
            return self.critic_target(state_s).view(-1)

    def polyak_update(self, target_network, network, polyak_factor=0.995):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(polyak_factor * param.data + target_param.data * (1.0 - polyak_factor))

    def update_soft(self,net_target,net,polyak=0.995):
        with torch.no_grad():
            for p, p_targ in zip(net.parameters(), net_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


if __name__ == "__main__":
    p="/home/eranhe/car_model/generalization/new/4"
    a = Agent(os.path.join(p,"critic.csv"))
    b = load_buffer(os.path.join(p,"data.csv"))
    for _ in range(100000):
        a.learn(buffer=b)
