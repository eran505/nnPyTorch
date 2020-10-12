from os.path import expanduser
import nn_pytorch as nnpy
import torch
import numpy as np
import csv,math
from preprocessor import Loader, RegressionFeature
from socket import gethostname


min_ = np.array([10.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 138.0, 140.0, 0.0, -1.0, -1.0, -1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0])
ptp_ = np.array([162.0, 167.0, 3.0, 300.0, 300.0, 3.0, 162.0, 300.0, 300.0, 3.0, 2.0, 2.0, 2.0, 162.0, 167.0, 3.0, 2.0, 2.0, 2.0,
        300.0, 300.0, 3.0, 162.0, 160.0, 3.0])


def norm(f):
    f_norm = (f - min_) / ptp_

    return f_norm


class schedulerAction(object):


    @staticmethod
    def get_move(diff):
        max_diff = max(diff)
        if max_diff==0:
            return 1
        b = math.ceil(math.log2(max_diff))
        b= max(b-3,0)
        action_number = pow(2,b)
        return action_number




class Transformer(object):

    def __init__(self, dir_data):
        self.loader = None
        self.reg = None
        self.loader_init(dir_data)
        self.distance_man = lambda vec_1, vec_2: np.absolute(vec_1 - vec_2)

    def loader_init(self, dir_data):
        self.loader = Loader(dir_data)
        self.loader.load_p("{}/p.csv".format(dir_data))
        self.loader.load_game_setting("{}/con.csv".format(dir_data))
        dico_info_game = self.loader.get_config_id(0)
        attacker_paths = self.loader.get_path_object()
        self.reg = RegressionFeature(attacker_paths, dico_info_game)

    def get_F(self, np_arr):
        return self.reg.get_F(np_arr)


class AgentA(object):

    def __init__(self, csv_path):
        self.all_paths = []
        self.w_paths = []
        self.get_all_paths(csv_path)
        self.path_indexes = np.arange(start=0, stop=len(self.w_paths), step=1)
        self.path_number = -1
        self.step_t = -1

    def get_all_paths(self, csv_all_paths):
        self.read_file(csv_all_paths)

    def read_file(self, path_file):
        all_p = []
        with open(path_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            for row in spamreader:
                if len(row) < 2:
                    continue
                self.w_paths.append(float(row[0].split(":")[-1]))
                arr = [list(eval(x[1:-2])) for x in row[1:] if len(x) > 5]
                l = np.array(arr)
                all_p.append(l)
        self.all_paths = all_p


    def next_move(self,num_repeated_action):
        self.step_t += num_repeated_action
        if self.step_t >= self.all_paths[self.path_number].shape[0]:
            self.step_t=self.all_paths[self.path_number].shape[0]-1
            #raise Exception("EndOfPathException")
        self.cur_state = self.all_paths[self.path_number][self.step_t, :]

    def reset(self):
        self.path_number = np.random.choice(self.path_indexes, 1, False)[0]#, p=self.w_paths)[0]
        self.step_t = 0
        self.cur_state = self.all_paths[self.path_number][self.step_t, :]

    def __str__(self):
        return "A_" + str([tuple(x) for x in self.cur_state])


class AgentD(object):

    def __init__(self, data_path, nn_path, start_pos, max_speed=1):
        self.max_speed = 1
        self.actions = None
        self.make_action_list()
        self.start_positions = start_pos
        self.nn = None
        self.load_nn(nn_path)
        self.cur_state = None
        self.reset()
        self.trans = Transformer(data_path)

    def load_nn(self, path_to_model):
        self.nn = nnpy.LR(25)
        self.nn.load_state_dict(torch.load(path_to_model,map_location=device))
        #self.nn = self.nn.double()
        self.nn.eval()

    def reset(self):
        self.cur_state = np.array(self.start_positions, copy=True)

    def get_move(self, pos_A):
        v = np.zeros(27)
        f = self.get_F_D(pos_A)
        f = np.hstack((f.flatten(), np.zeros(3))).ravel()
        for i in range(27):
            f[-3:] = self.actions[i]
            expected_reward_y = self.nn(torch.tensor(norm(f)).double())
            v[i] = expected_reward_y
            print("{}:->{}".format(i,v[i]))
        print("np.argmax = {} ".format(np.argmax(v)))
        exit()
        return np.argmax(v)

    def get_move_all(self, pos_A):
        f = self.get_F_D(pos_A)
        f = np.hstack((f.flatten())).ravel()
        f = f.astype('f')
        #print(len(f))
        expected_reward_y = self.nn(torch.tensor(norm(f)).unsqueeze(0).float()) #.double()
        #print(expected_reward_y)
        arg_max_action = np.argmax(expected_reward_y.detach().numpy())
        #print("np.argmax = {} ".format(arg_max_action))
        return arg_max_action

    def get_F_D(self, posA):
        a = np.array([posA.flatten(), self.cur_state.flatten()]).flatten()
        a = np.expand_dims(a,axis=0)
        return self.trans.get_F(a)


    def next_move(self, pos_A,num_repeated_action):
        action_a_id = self.get_move_all(pos_A)
        #action_a_id=1
        for _ in range(num_repeated_action):
            self.apply_action(action_a_id)

    def apply_action(self, action_id):
        action_a = self.actions[action_id]
        speed = self.cur_state[1, :]
        new_speed = action_a + speed
        new_speed[new_speed > self.max_speed] = self.max_speed
        new_speed[new_speed < -self.max_speed] = -self.max_speed
        self.cur_state[0, :] = new_speed + self.cur_state[0, :]
        self.cur_state[1, :] = new_speed

    def make_action_list(self):
        l = []
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                for z in range(-1, 2, 1):
                    l.append(np.array([x, y, z]))
        self.actions = np.array(l)

    def __str__(self):
        return "D_" + str([tuple(x) for x in self.cur_state])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Game(object):

    def __init__(self, dir_data, dir_nn, num=11):
        self.home = expanduser("~")
        self.pow2diffs=[np.array([pow(2,3+x),pow(2,3+x),4]) for x in range(12)]
        self.grid_size = None
        self.golas = None
        self.d_setting=None
        self.D = None
        self.A = None
        self.get_info_game(dir_data)
        self.info = np.zeros(3)
        self.construct(dir_data, dir_nn, num)

    def get_info_game(self, dir_data):
        obj = Loader(dir_data)
        obj.load_game_setting("{}/con.csv".format(dir_data))
        d = obj.get_config_id(0)
        self.grid_size = np.array([d['X'],d['Y'],d['Z']])
        self.golas=d['P_G']
        self.d_setting=d



    def construct(self, dir_data, dir_nn, num=2):
        self.A = AgentA("{}/p.csv".format(dir_data))
        self.D = AgentD(dir_data, "{}/nn{}.pt".format(dir_nn, num),
                        np.array([self.d_setting['D_start'].squeeze(0), np.zeros(3)]))

    def main_loop(self, max_iter):
        for _ in range(max_iter):
            self.A.reset()
            self.D.reset()
            while True:
                num_actions = schedulerAction.get_move(np.abs(self.A.cur_state[0,:]-self.D.cur_state[0,:]))
                #self.print_state(num_actions)
                if self.mini_game_end():
                    #print("END")
                    break
                self.D.next_move(self.A.cur_state,num_actions)
                self.A.next_move(num_actions)

        self.print_info()

    def mini_game_end(self):
        if self.if_A_at_goal(self.A.cur_state[0, :]):
            self.info[0] += 1
            return True
        if np.any(self.grid_size <= self.D.cur_state[0, :]) or np.any(self.D.cur_state[0, :] < 0):
            self.info[1] += 1
            return True
        if np.all(self.D.cur_state[0, :] == self.A.cur_state[0, :]):
            self.info[2] += 1
            return True

    def print_info(self):
        print("Goal:{}\tWall:{}\tCollision:{}".format(self.info[0], self.info[1], self.info[2]))

    def if_A_at_goal(self, pos_A):
        for item_goal in self.golas:
            if np.array_equal(pos_A, item_goal):
                return True
        return False

    def print_state(self,num_actions):
        print("{}|{} [A]:{}".format(str(self.A), str(self.D),num_actions))


if __name__ == "__main__":
    l = []

    home = expanduser("~")

    data_path = "{}/car_model/generalization/3data".format(home)
    nn_path = "{}/car_model/nn".format(home)

    for i in range(16,28):
        g = Game(data_path, nn_path, i)
        g.main_loop(100)
        l.append(g.info[2])
    x = np.argmax(np.array(l))

