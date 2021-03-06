import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Used for Box2D / Toy problems


class FC_Q(nn.Module):
    def __init__(self, num_actions,state_dim):
        super(FC_Q, self).__init__()
        self.q1 = nn.Linear(state_dim, 256)
        self.q2 = nn.Linear(256, 256)
        self.q3 = nn.Linear(256, num_actions)

        self.i1 = nn.Linear(state_dim, 256)
        self.i2 = nn.Linear(256, 256)
        self.i3 = nn.Linear(256, num_actions)

    def forward(self, state):

        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))
        q = self.q3(q)

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))

        return q,F.log_softmax(i, dim=1), i


class discrete_BCQ(object):
    def __init__(
            self,
            num_actions,
            state_dim,
            device,
            BCQ_threshold=0.3,
            discount=0.987,
            optimizer="Adam",
            optimizer_parameters={},
            polyak_target_update=False,
            target_update_frequency=8e3,
            tau=0.005,
            initial_eps=1,
            end_eps=0.001,
            eps_decay_period=25e4,
            eval_eps=0.001,
    ):

        self.device = device

        # Determine network type
        self.Q = FC_Q(num_actions,state_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Threshold for "unlikely" actions
        self.threshold = BCQ_threshold

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state_s, eval=False):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        state_s = torch.tensor(state_s, dtype=torch.float).to(self.device)
        state_s = torch.squeeze(state_s)
        with torch.no_grad():
            state = torch.FloatTensor(state_s).reshape(self.state_shape).to(self.device)
            q, imt, i = self.Q(state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()
            # Use large negative number to mask actions from argmax
            return int((imt * q + (1. - imt) * -1e8).argmax(1))


    def train(self, replay_buffer):
        # Sample replay buffer
        batch=2
        state, action, reward, next_state, done = replay_buffer.sample(batch)

        done = torch.tensor(done,dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int64).resize(batch,1)
        reward = torch.tensor(reward, dtype=torch.float)


        # Compute the target Q value
        with torch.no_grad():
            next_state = torch.tensor(next_state,dtype=torch.float).to(self.device)
            next_state = torch.squeeze(next_state)
            q, imt, i = self.Q(next_state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()
            #print(imt)
            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

            q, imt, i = self.Q_target(next_state)
            q_max = torch.squeeze( q.gather(1, next_action))
            target_Q = reward + (1-done) * self.discount * q_max#.reshape(-1, 1)

        # Get current Q estimate
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = torch.squeeze(state)

        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action)

        # Compute Q loss
        q_loss = F.smooth_l1_loss(torch.squeeze(current_Q), target_Q)
        i_loss = F.nll_loss(imt, action.reshape(-1))

        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save_model(self,file_path_name):
        torch.save(self.Q.state_dict(), file_path_name)


if __name__ == '__main__':
    pass