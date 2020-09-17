from os.path import expanduser
import nn_pytorch as nnpy
import torch
import numba as nb
import numpy as np
import csv

min = np.array([1., 0., 0., 0., 0., - 1., 9., 9., 0., - 1., - 1., - 1., - 1., - 1., - 1.])
ptp = np.array([17., 19., 2., 2., 2., 2., 11., 11., 3., 2., 2., 2., 2., 2., 2.])

def norm(f):
    f_norm = (f - min) / ptp

    return f_norm

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
                self.w_paths.append(float(row[0].split(":")[-1]))
                l = np.array(list(map(lambda x: np.array(eval(x[1:-2])), row[1:])))
                all_p.append(l)
        self.all_paths = np.array(all_p)

    def next_move(self):
        self.step_t += 1
        if (self.step_t >= self.all_paths[self.path_number].shape[0]):
            raise Exception("EndOfPathException")
        self.cur_state = self.all_paths[self.path_number, self.step_t, :]

    def reset(self):
        self.path_number = np.random.choice(self.path_indexes, 1, False, p=self.w_paths)[0]
        self.step_t = 0
        self.cur_state = self.all_paths[self.path_number, self.step_t, :]

    def __str__(self):
        return "A_" + str([tuple(x) for x in self.cur_state])


class AgentD(object):

    def __init__(self, nn_path, start_pos, max_speed=1):
        self.max_speed = 1
        self.actions = None
        self.make_action_list()
        self.start_positions = start_pos
        self.nn = None
        self.load_nn(nn_path)
        self.cur_state = None
        self.reset()


    def load_nn(self, path_to_model):
        self.nn = nnpy.LR(35)
        self.nn.load_state_dict(torch.load(path_to_model))
        self.nn = self.nn.double()
        self.nn.eval()

    def reset(self):
        self.cur_state = np.array(self.start_positions, copy=True)

    def get_move(self, pos_A):
        v = np.zeros(27)
        f = np.hstack((pos_A.flatten(), self.cur_state.flatten(), np.zeros(3))).ravel()
        for i in range(27):
            f[-3:] = self.actions[i]
            #print(f)
            expected_reward_y = self.nn(torch.tensor(norm(f)).double())
            v[i] = expected_reward_y
        #print(v)
        return np.argmax(v)

    def next_move(self, pos_A):
        action_a_id = self.get_move(pos_A)
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

    def __init__(self,num=11):
        self.home = expanduser("~")
        self.grid_size = np.array([21, 21, 4])
        self.golas = [np.array([16, 20, 0]),np.array([19, 15, 0]) ]
        self.D = None
        self.A = None
        self.info = np.zeros(3)
        self.construct(num)

    def construct(self,num=2):
        self.A = AgentA("{}/car_model/generalization/data/p.csv".format(self.home))
        self.D = AgentD("/home/ERANHER/car_model/nn/nn{}.pt".format(num), np.array([[20, 20, 0], [0, 0, 0]]), max_speed=1)

    def main_loop(self, max_iter):
        for i in range(max_iter):
            self.A.reset()
            self.D.reset()
            while True:
                #self.print_state()
                if self.mini_game_end():
                    #print("END")
                    break
                self.D.next_move(self.A.cur_state)
                self.A.next_move()

        self.print_info()


    def mini_game_end(self):
        if self.if_A_at_goal(self.A.cur_state[0, :]):
            self.info[0] += 1
            return True
        if np.any(self.grid_size <= self.D.cur_state[0, :]) or np.any(self.D.cur_state[0, :]<0) :
            self.info[1] += 1
            return True
        if np.all(self.D.cur_state[0, :] == self.A.cur_state[0, :]):
            self.info[2] += 1
            return True

    def print_info(self):
        print("Goal:{}\tWall:{}\tCollision:{}".format(self.info[0], self.info[1], self.info[2]))

    def if_A_at_goal(self,pos_A):
        for item_goal in self.golas:
            if np.array_equal(pos_A,item_goal):
                return True
        return False

    def print_state(self):
        print("{}|{}".format(str(self.A),str(self.D)))


if __name__ == "__main__":
    l=[]
    for i in range(55):
        g = Game(i)
        g.main_loop(20)
        l.append(g.info[2])
    x = np.argmax(np.array(l))
    print(x)

