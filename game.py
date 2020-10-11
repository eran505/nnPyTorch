from os.path import expanduser
import nn_pytorch as nnpy
import torch
import numpy as np
import csv
from preprocessor import Loader, RegressionFeature
from socket import gethostname

min = np.array([4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0, 8.0, 0.0, -1.0, -1.0, -1.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
ptp = np.array([7.0, 11.0, 3.0, 8.0, 19.0, 3.0, 11.0, 5.385164807134504, 6.0, 17.0, 3.0, 4.0, 2.0, 2.0, 7.0, 11.0, 3.0, 2.0, 2.0, 2.0, 6.0, 17.0, 3.0, 6.0, 17.0, 3.0, 3.0, 17.0, 3.0, 7.0, 11.0, 3.0, 7.0, 11.0, 3.0, 5.0, 11.0, 3.0, 2.0, 2.0, 2.0])

min_ = np.array([4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0, 8.0, 0.0, -1.0, -1.0, -1.0, 2.0,
        2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ptp_ = np.array([7.0, 11.0, 3.0, 8.0, 19.0, 3.0, 11.0, 5.385164807134504, 6.0, 17.0, 3.0, 4.0, 2.0, 2.0, 7.0, 11.0, 3.0, 2.0,
        2.0, 2.0, 6.0, 17.0, 3.0, 6.0, 17.0, 3.0, 3.0, 17.0, 3.0, 7.0, 11.0, 3.0, 7.0, 11.0, 3.0, 5.0, 11.0, 3.0])


def norm(f):
    f_norm = (f - min_) / ptp_

    return f_norm


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


    def next_move(self):
        self.step_t += 1
        if (self.step_t >= self.all_paths[self.path_number].shape[0]):
            raise Exception("EndOfPathException")
        self.cur_state = self.all_paths[self.path_number][self.step_t, :]

    def reset(self):
        self.path_number = np.random.choice(self.path_indexes, 1, False, p=self.w_paths)[0]
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
        self.nn = nnpy.LR(38)
        self.nn.load_state_dict(torch.load(path_to_model))
        self.nn = self.nn.double()
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
        #print(len(f))
        expected_reward_y = self.nn(torch.tensor(norm(f)).double())
        #print(expected_reward_y)
        arg_max_action = np.argmax(expected_reward_y.detach().numpy())
        #print("np.argmax = {} ".format(arg_max_action))
        return arg_max_action

    def get_F_D(self, posA):
        a = np.array([posA.flatten(), self.cur_state.flatten()]).flatten()
        a = np.expand_dims(a,axis=0)
        return self.trans.get_F(a)


    def next_move(self, pos_A):
        action_a_id = self.get_move_all(pos_A)
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


class Game(object):

    def __init__(self, dir_data, dir_nn, num=11):
        self.home = expanduser("~")

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
                #self.print_state()
                if self.mini_game_end():
                    # print("END")
                    break
                self.D.next_move(self.A.cur_state)
                self.A.next_move()

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

    def print_state(self):
        print("{}|{}".format(str(self.A), str(self.D)))


if __name__ == "__main__":
    l = []

    data_path = "/home/eranhe/car_model/generalization/3data"
    nn_path = "/home/eranhe/car_model/nn"
    if (gethostname() == 'lab2'):
        data_path = "/home/lab2/eranher/car_model/generalization/3data"
        nn_path = "/home/lab2/eranher/car_model/nn"
    for i in range(55):
        g = Game(data_path, nn_path, i)
        g.main_loop(20)
        l.append(g.info[2])
    x = np.argmax(np.array(l))

