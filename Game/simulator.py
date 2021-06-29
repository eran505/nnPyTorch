import numpy as np
from math import log2

from preprocessor import Loader
import csv
from os import path
from copy import deepcopy

np.random.seed(1234)


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
        print(dico_info_game)
        print(attacker_paths)

    def get_F(self, np_arr):
        return self.reg.get_F(np_arr)


class schedulerAction(object):

    @staticmethod
    def get_move(diff):
        max_diff = max(diff)
        if max_diff == 0:
            return 1
        b = int(log2(max_diff)) + 1

        b = max(b - 3, 0)
        action_number = pow(2, b)

        return action_number


def get_action_dict():
    d={0:np.array([0,1,0]),
       1:np.array([1,0,0]),
       2:np.array([0,-1,0]),
       3:np.array([-1,0,0]),
       4:np.array([0,0,0])
       # 5: np.array([0, 0, -1]),
       # 6: np.array([0, 0, 1])
       }
    return d

def get_random_samples(num,Game_Sim_obj):
    l=[]
    for index_i in range(num):
        print(index_i)
        s = Game_Sim_obj.reset()
        done=False
        while not done:
            l.append(s)
            observation_, reward, done,info = Game_Sim_obj.step(np.random.randint(0,2))
            s=observation_
            if done:
                break
    return np.array(l)

import gym
from gym import spaces
class Game_Sim(gym.Env):

    def __init__(self, csv_dir):
        super(Game_Sim, self).__init__()
        metadata = {'render.modes': ['human']}
        self.inital_state = np.zeros(12,dtype=float)
        self.ctr_round = 0
        self.all_paths = []
        self.w_paths = []
        self.bound=None
        self.game_setting_dict = None
        self._get_all_paths(csv_dir)
        self.path_indexes = np.arange(start=0, stop=len(self.w_paths), step=1)
        self.path_number = -1
        self.step_t = 0
        self.state=None
        self.d_action=get_action_dict()
        self.ctr_index=0
        #self.action_space=len(get_action_dict())
        self.action_space=spaces.Discrete(len(get_action_dict()))



        # In the discrete case, the agent act on the binary
        # representation of the observation
        self.observation_space = spaces.Box(np.array([-1,-1,-1,-3,-3,-3,-1,-1,-1,-2,-2,-2,0,0,0,0,0,0,0,0,0]),
                                            np.array([15,15,15,3,3,3,15,15,15,2,2,2,4,4,4,4,4,4,15,15,15]))

    def expand_state(self,state):
        dif = state[:3]-state[6:9]
        powerA = state[3:6]**2
        powerD = state[9:12] ** 2
        merge_arr = np.concatenate([state,powerA,powerD,dif], axis=0)
        return merge_arr

    def get_norm_vector(self):
        a = np.zeros(12)
        for i in range(3):
            a[i]=self.bound[i]
            a[i+6] = self.bound[i]
        a[3:6] = 3
        a[3:6] = 2
        return a

    def _get_all_paths(self, csv_dir,add_sub_path=False):
        self._read_file(path.join(csv_dir,'p.csv'))
        obj = Loader(None)
        obj.load_game_setting(path.join(csv_dir,'con.csv'))
        self.game_setting_dict = obj.d_conf
        self.bound = np.array([self.game_setting_dict["X"][0],
                               self.game_setting_dict["Y"][0],
                               self.game_setting_dict["Z"][0]])
        print(self.game_setting_dict)
        self._set_init_state()
        l=[]
        if add_sub_path:
            l.append(self.all_paths[0][:,:])
            l.append(self.all_paths[0][1:,:])
            l.append(self.all_paths[0][2:,:])
            self.all_paths=l
            print("")

    def render(self, mode='human'):
        print("No render")

    def close(self):
        print("END")

    def _set_init_state(self):
        posA = [ eval(x) for x in str(self.game_setting_dict["A_start"][0]).split('|')]
        posD = [ eval(x) for x in str(self.game_setting_dict["D_start"][0]).split('|')]
        for idx,item in enumerate(posA):
            self.inital_state[idx]=item
        for idx, item in enumerate(posD):
            self.inital_state[idx+6] = item

    def _read_file(self, path_file):
        all_p = []
        with open(path_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            for row in spamreader:
                if len(row) < 2:
                    continue
                self.w_paths.append(float(row[0].split(":")[-1]))
                arr = [np.array(eval(x[1:-2])) for x in row[1:] if len(x) > 5]
                l = np.array(arr)
                l = l.reshape(len(l),6)
                all_p.append(l)
        self.all_paths = all_p

    def _next_move(self):
        done = False
        self.step_t += 1

        if self.step_t >= self.all_paths[self.path_number].shape[0]:
            self.step_t = self.all_paths[self.path_number].shape[0] - 1
            done = True
        self.state[:6] = self.all_paths[self.path_number][self.step_t, :]
        return done

    def reset(self):

        self.path_number = np.random.choice(self.path_indexes, 1, False)[0]

        self.ctr_round += 1
        self.path_number = self.ctr_round % len(self.path_indexes)
        self.step_t = 0
        self.state = deepcopy(self.inital_state)
        return self.expand_state(self.state)

    def on_attacker_path(self):
        d_pos = self.state[6:9]
        sub_path = self.all_paths[self.path_number][self.step_t:]
        for idx,item in enumerate(sub_path):
            if item == d_pos:
                return True,idx
        return False,-1

    def _check_for_end_game(self,sub_reward=False):

        if (self.state[:3] == self.state[6:9]).all():
            print("Coll","P: ",self.path_number," \tloc:",self.state[:3])
            r=1
            # if self.path_number==1:
            #     r=30
            return True,r,'C'
        if (0 > self.state[6:9]).any():
            return True,-2,'w'
        if (self.bound<=self.state[6:9]).any():
            return True,-2,'w'

        if sub_reward:
            bol,idx=self.on_attacker_path()
            if bol:
                return False,0.03,'o'

        return False,0,'-'

    def _make_action(self,a):
        self.state[-3:]+=a
        for i in range(1,4,1):
            if self.state[-i]>1:
                self.state[-i]=1
            elif self.state[-i]<-1:
                self.state[-i]=-1
        self.state[6:9]+=self.state[-3:]


    def step(self,a):
        a = int(a)
        self._make_action(self.d_action[a])
        if(self._next_move()):
            return self.state,-1.5,True,{"episode":None,"is_success":None}
        done,r,info = self._check_for_end_game()
        #print(self.state)
        return self.expand_state(self.state),r,done,{"episode":None,"is_success":None}


if __name__ == '__main__':

    g = Game_Sim("/home/eranhe/car_model/debug")
    for _ in range(4):
        print("---NEW GAME---")
        done = False
        state = g.reset()
        while not done:
            print(state)
            state,r,done,info = g.step(4)
