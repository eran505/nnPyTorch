
import argparse
import copy
import importlib
import json
import os
import matplotlib.pyplot as plt

from os.path import expanduser
import numpy as np
import torch
from RL.BCQ import discrete_BCQ
import pandas as pd
from PRB.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from batch_learning import get_data

import RL.static_dqn as algo_static
def load_buffer(p="/home/eranhe/car_model/debug/data.csv"):
    size = 25000
    my_replay_buffer = ReplayBuffer(size)
    data = pd.read_csv(p)
    print(list(data))
    data.iloc[:,12]=data.iloc[:,12]*26
    data = data.sample(frac=1)
    print(len(data))
    data = data.to_numpy()
    for i, item in enumerate(data):
        my_replay_buffer.push(item[:12], item[12], item[25], item[13:25], item[26])
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
        actionz_pred =int(actionz_pred)
        correct += (actionz_pred == int(action[0]))
    print("test - correct:\t", correct / float(iter_number))
    return correct / float(iter_number)


def algo():
    test = load_buffer("/home/eranhe/car_model/debug/df_test.csv")
    agent = algo_static.BCQ(12,1,27)
    replay_b = load_buffer("/home/eranhe/car_model/debug/df_test.csv")
    #replay_b = load_buffer()
    for index in range(1000):
        agent.train(replay_b,100,1)
        eval_test(test,agent)

if __name__ == '__main__':
    algo()