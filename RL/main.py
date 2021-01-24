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


def plot_loss(x, array, dist):
    plt.plot(x, array)
    plt.savefig('{}'.format(dist))  # save the figure to file
    plt.show()


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


def train_BCQ(replay_buffer, num_actions, state_dim, device, args, parameters):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize and load policy
    policy = discrete_BCQ(
        num_actions,
        state_dim,
        device,
        args.BCQ_threshold,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
    )

    # Load replay buffer

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    test = load_buffer("/home/eranhe/car_model/debug/df_test.csv")
    h = []

    while training_iters < args.max_timesteps:

        for k in range(int(parameters["eval_freq"])):
            if k % 4e3 == 0:
                print("iter=", k)
                policy.save_model("{}/car_model/nn/nn{}.pt".format(expanduser("~"), int(training_iters / 4e3)))
                h.append(eval_test(test_buffer=test, policy=policy))
            policy.train(replay_buffer)


        training_iters += int(parameters["eval_freq"])
        print(f"Training iterations: {training_iters}")
    x = [k * 4e3 for k in range(len(h))]
    plot_loss(x, h, "/home/eranhe/car_model/debug/fig.png")


if __name__ == "__main__":
    regular_parameters = {
        # Exploration
        "start_timesteps": 1e3,
        "initial_eps": 0.1,
        "end_eps": 0.1,
        "eps_decay_period": 1,
        # Evaluation
        "eval_freq": 5e3,
        "eval_eps": 0,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 64,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-3
        },
        "train_freq": 1,
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005
    }
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PongNoFrameskip-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=21365, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Default")  # Prepends name to filename
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--BCQ_threshold", default=0.3, type=float)  # Threshold hyper-parameter for BCQ
    parser.add_argument("--low_noise_p", default=0.2,
                        type=float)  # Probability of a low noise episode when generating buffer
    parser.add_argument("--rand_action_p", default=0.2,
                        type=float)  # Probability of taking a random action when generating buffer, during non-low noise episode
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral policy
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize buffer
    replay_buffer = load_buffer("/home/eranhe/car_model/debug/data.csv")
    num_actions = 27
    state_dim = 12
    parameters = regular_parameters
    train_BCQ(replay_buffer, num_actions, state_dim, device, args, parameters)
