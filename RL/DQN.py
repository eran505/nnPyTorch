import torch
from  os.path import expanduser
from torch import nn
import PRB.replay_buffer
import numpy as np
from batch_learning import get_data
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions,hidden=128):
        super(DQN, self).__init__()
        self.num_of_action=num_actions
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),

        )

    def forward(self, x):
        return self.layers(x)



class Agent(object):

    def __init__(self,num_in,num_out,buffer_size,lr):
        self.target_model=DQN(num_in,num_out)
        self.current_model = DQN(num_in,num_out)
        self.replay_buffer = PRB.replay_buffer.ReplayBuffer(buffer_size)
        self.optimizer = torch.optim.Adam(lr=lr,params=self.current_model.parameters())
        self.gamma=0.987
        self.target_model.to(device)
        self.current_model.to(device)
        self.batch_size=1
        self.beta=0.5
        self.transform = None
        self.w=None
        self.load_buffer()
        self.home = expanduser("~")

    def load_buffer(self):
        df = pd.read_csv("/home/eranhe/car_model/debug/data.csv")
        data = df.to_numpy()
        for item in data:
            #print(len(item))
            self.replay_buffer.push(item[:12],item[12]*26,item[25],item[13:25],item[26])

    def compute_td_loss(self,batch_size, beta):
        state, action, reward, next_state, done  = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        #weights = torch.FloatTensor(weights)
        #print(action.item())

        state = state.to(device)
        next_state = next_state.to(device)

        with torch.no_grad():
            next_q_values = self.target_model(next_state).squeeze(1)
        q_values = self.current_model(state).squeeze(1)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2) #* weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()

        #print("loss: ",loss.item())
        for param in self.current_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        #self.replay_buffer.update_priorities(indices,prios.detach().numpy())

        self.update_soft(self.target_model, self.current_model)

        return loss

    def learn(self):
        steps = 0
        while steps < 10000000:
            self.compute_td_loss(self.batch_size,self.beta)
            steps=steps+1
            if steps%10000==0:
                print("check-point {}".format(int(steps/1000)))
                torch.save(self.current_model.state_dict(), "{}/car_model/nn/nn{}.pt".format(self.home, int(steps/1000)))

    def update_soft(self,net_target,net,polyak=0.995):
        with torch.no_grad():
            for p, p_targ in zip(net.parameters(), net_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


def load_dir(dir_data):
    return

if __name__ == '__main__':
    p=""
    a = Agent(12,27,100000,0.03)
    a.learn()