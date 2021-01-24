
from abc import ABC
import numpy as np
from Game.simulator import Game_Sim,get_random_samples
from os import getcwd
from os import path

from sklearn import preprocessing, pipeline
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt

from torch import nn
import torch
import random
import torch.nn.functional as F

random.seed(123456)
torch.manual_seed(123456)

cwd = getcwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
print('Using device:', device)
# Additional Info when using cuda
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')





class RBF(object):

    def __init__(self):
        self.featurizer = None
        self.scaler = None
        self.num_of_c = 10
        self.num_rbf = 4
        self.init_rbf()

    def get_num_f(self):
        return self.num_rbf*self.num_of_c

    def get_norm(self):
        observation_examples = np.array(get_random_samples(20000))
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        self.featurizer.fit(self.scaler.transform(observation_examples))

    def init_rbf(self):
        observation_examples = np.array(get_random_samples(10000))
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        self.featurizer = pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=self.num_of_c)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=self.num_of_c)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=self.num_of_c)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=self.num_of_c))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

    def featurize_state(self, state):
        # Transform data
        scaled = self.scaler.transform(state.reshape(1,-1))
        featurized = self.featurizer.transform(scaled)
        return featurized






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
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.5)
        self.to(device)
        self.epslion = 0.00
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
        # if random.random() < self.epslion:
        #     print("at rand")
        #     return random.randrange(3)
        action_probs = torch.distributions.Categorical(actions_probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        return action.item(), log_prob

    def sample_policy(self, state,eval=False):
        pi = self.forward(state)
        probabilities = torch.distributions.Categorical(pi)
        if random.random()<self.epslion and not eval:
            actions=torch.randint(3, (1,))
        else:
            actions = probabilities.sample()
        log_probs = probabilities.log_prob(actions)
        return actions, log_probs

    def save(self,suffix=""):
        torch.save(self.state_dict(), self.checkpoint_file+str(suffix)+".pt")

    def load_checkpoint(self,path=None):
        if path is None:
            self.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.load_state_dict(torch.load(path))

class CriticNet(nn.Module, ABC):
    def __init__(self, input_dims, fc1_dim, fc2_dim, beta, out_dim=1):
        super(CriticNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dim
        self.fc2_dims = fc2_dim
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, out_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.5)
        self.to(device)
        self.checkpoint_file = path.join(cwd, "modelz/criticNet")

    def forward(self, observation):
        observation = torch.squeeze(observation)
        x = F.relu_(self.fc1(observation))
        x = F.relu_(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self,suffix=""):
        torch.save(self.state_dict(), self.checkpoint_file+str(suffix)+".pt")

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, env,_norm_vec):
        self.Rbf =RBF()
        self.norm_vec = _norm_vec
        self.gamma = 1.0
        self.env = env
        self.log_prob = None
        self.features_number = self.Rbf.get_num_f()
        self.actor = None
        self.critic = None
        self.critic_target = None
        self.learn_at = 100
        self.memo = None
        self.ctr = 0
        self.update_target_estimator_every = 1000
        self.init_networks()
        self.eval=False


    def get_learning_steps(self):
        return self.ctr

    def init_networks(self):
        self.actor = ActorNet(input_dims=self.features_number, fc1_dim=256, fc2_dim=256, action_num=5, alpha=0.0001)
        self.critic = CriticNet(input_dims=self.features_number, fc1_dim=256, fc2_dim=256, beta=0.0001)
        self.critic_target = CriticNet(input_dims=self.features_number, fc1_dim=256, fc2_dim=256, beta=0.0001)

    def learn(self):
        self.ctr += 1

        self.critic.zero_grad()


        state, action, reward, new_state, done = self.memo
        reward = torch.tensor(reward).float().to(device)
        done = torch.tensor(done).float().to(device)
        state_next = torch.tensor(new_state).float().to(device)
        state = torch.tensor(state).float().to(device)

        value = self.critic.forward(state).view(-1)
        value_next = self.get_target_val(state_next)
        #value_next = self.critic(state_next)


        advantage = reward + (1.0 - done) * self.gamma * value_next - value

        critic_loss = advantage ** 2

        actor_loss = -self.log_prob * advantage.detach()

        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()


        self.update_soft(self.critic_target,self.critic)


    def load_actor(self,path_model):
        self.actor.load(path_model)

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def take_step_scheduler(self):
        print("[{}]".format("take_step_scheduler"))
        self.actor.scheduler.step()
        self.critic.scheduler.step()

    def choose_action(self, observation):
        f_observation = self.Rbf.featurize_state(observation)
        f_observation = torch.from_numpy(f_observation).float().to(device)
        actions ,log_prob = self.actor.sample_policy(f_observation,self.eval)
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


    def update_soft(self,net_target,net,polyak=0.9): #995
        with torch.no_grad():
            for p, p_targ in zip(net.parameters(), net_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)



def main(dir_p):
    env = Game_Sim(dir_p)
    norm_vector = env.get_norm_vector()
    agent = Agent(20,norm_vector)
    total_steps=0

    num_of_iter=2000

    l_r = np.zeros(num_of_iter)
    env.reset()
    for j in range(num_of_iter):
        score = 0
        done = False
        observation = env.reset()
        action_list=[]
        info=""
        while not done:
            action = agent.choose_action(observation)
            action_list.append(action)
            observation_, reward, done,info = env.step(action)
            score += reward
            agent.save_buffer(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            total_steps += 1
        print("score(",j,"):\t",score,"--",info,"\t\tA:",action_list)
        l_r[j]=score


    plt.plot([x for x in range(num_of_iter)],l_r)
    plt.show()
if __name__ == '__main__':
    main("/home/eranhe/car_model/debug")
    pass