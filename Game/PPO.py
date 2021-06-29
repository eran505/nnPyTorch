
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from Game.simulator import  Game_Sim
import matplotlib.pyplot as plt
import numpy as np
import gym
from Game.leaner import RBF
SEED = 20205

np.random.seed(SEED)
torch.manual_seed(SEED)

train_env = Game_Sim("/home/eranhe/car_model/debug")
test_env =  Game_Sim("/home/eranhe/car_model/debug")



rbf = RBF(train_env)
#self.rbf.featurize_state(observation)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        #print("input==="*10)
        #print(x)
        x = self.net(x)
        #print("out==="*10)
        #print(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)

        return action_pred, value_pred



INPUT_DIM = 4
HIDDEN_DIM = 128
OUTPUT_DIM = 2
EP=0.00001

actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

policy = ActorCritic(actor, critic)



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


policy.apply(init_weights)
LEARNING_RATE = 0.0001

optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)


def train(env, policy, optimizer, discount_factor, ppo_steps, ppo_clip):
    policy.train()

    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()
    state = rbf.featurize_state(state)
    info_array=[]
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)

        # append state here, not after we get the next state from env.step()
        states.append(state)

        action_pred, value_pred = policy(state)

        action_prob = F.softmax(action_pred, dim=-1)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()

        # if np.random.random()<EP:
        #     action[0] = np.random.randint(0,4)

        log_prob_action = dist.log_prob(action)

        state, reward, done, info = env.step(action.item())
        #print("done:\t",done,"\tinfo:\t",info,"\tR:",reward)


        info_array.append(info)

        state = rbf.featurize_state(state)

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward

    #print(info_array)
    states = torch.cat(states)
    actions = torch.cat(actions)
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer,
                                            ppo_steps, ppo_clip)

    #print(actions.numpy().flatten())

    return policy_loss, value_loss, episode_reward


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if len(rewards)==1:
        return returns.squeeze(-1)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()


    return returns


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values

    if normalize and len(advantages)>1:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    total_policy_loss = 0
    total_value_loss = 0

    states = states.detach()
    actions = actions.detach()
    log_prob_actions = log_prob_actions.detach()
    advantages = advantages.detach()
    returns = returns.detach()


    for _ in range(ppo_steps):
        # get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)



        # new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)

        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()

        policy_loss_1 = policy_ratio * advantages

        policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - ppo_clip, max=1.0 + ppo_clip) * advantages

        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

        value_pred = value_pred.squeeze()

        value_loss = F.smooth_l1_loss(returns, value_pred).mean()
        #print("policy_loss:",policy_loss)
        #print("value_loss:",value_loss)

        optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def evaluate(env, policy):
    policy.eval()

    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()
    state = rbf.featurize_state(state)
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim=-1)

        action = torch.argmax(action_prob, dim=-1)

        state, reward, done, info = env.step(action.item())
        state = rbf.featurize_state(state)
        episode_reward += reward

    return episode_reward


MAX_EPISODES = 1_000
DISCOUNT_FACTOR = 1.0
N_TRIALS = 3
REWARD_THRESHOLD = 1000
PRINT_EVERY = 10
PPO_STEPS = 1
PPO_CLIP = 0.2

train_rewards = []
test_rewards = []
train_rewards_eval=[]

for episode in range(1, MAX_EPISODES + 1):

    policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)

    test_reward = evaluate(test_env, policy)



    train_rewards.append(train_reward)
    test_rewards.append(test_reward)
    #train_rewards_eval.append(evaluate(train_env, policy))

    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
    #mean_train_rewards_eval = np.mean(train_rewards_eval[-N_TRIALS:])

    if episode % PRINT_EVERY == 0:
        print(
            f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} ')
    if mean_test_rewards >= REWARD_THRESHOLD:
        print(f'Reached reward threshold in {episode} episodes')

        break



plt.figure(figsize=(12,8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
plt.legend(loc='lower right')
plt.grid()
plt.show()