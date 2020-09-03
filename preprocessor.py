import pandas as pd
import numpy as np
import csv
from os.path import expanduser
from collections import Counter


def process_path(path_str):
    return [eval(x[1:-2]) for x in path_str]


class FeatureOperation(object):
    pass


class Loader(object):
    def __init__(self, father_dir):
        self.d_attacker_path = []
        self.father_path_dir = father_dir
        self.d_conf = None

    def load_p(self, p):
        with open(p, "r") as f:
            reader = csv.reader(f, delimiter=";")
            for i, line in enumerate(reader):
                def pro(line):
                    d = {'p': float(str(line[0]).split(':')[-1]), 'traj': process_path(line[1:])}
                    return d

                self.d_attacker_path.append(pro(line))

    def get_config_id(self, id_number):
        d_result = {}
        for ky in self.d_conf:
            if str(self.d_conf[ky][id_number]).__contains__('|'):
                l = []
                arr = str(self.d_conf[ky][id_number]).split('-')
                for item in arr:
                    item = item.split('|')
                    item = [int(j) for j in item]
                    np_arr = np.array((item[0], item[1], item[2]))
                    l.append(np_arr)
                d_result[ky] = np.array(l)
            elif str(self.d_conf[ky][id_number]).__contains__('-'):
                arr = str(self.d_conf[ky][id_number]).split('-')
                arr = [int(i) for i in arr]
                d_result[ky] = np.array(arr)
            else:
                d_result[ky] = self.d_conf[ky][id_number]
        d_result['grid_size']=np.array([d_result['X'],d_result['Y'],d_result['Z']])
        return d_result

    def load_game_setting(self, csv_p):
        df = pd.read_csv(csv_p)
        self.d_conf = df.to_dict()

    def get_path_object(self):
        return AttackerPaths(self.d_attacker_path)


class AttackerPaths(object):

    def __init__(self, d_pathz):
        self.path_info = d_pathz
        self.to_np_arry()
        self.distance_eq = lambda vec_1, vec_2: np.sqrt(np.sum((vec_1 - vec_2) ** 2))
        self.distance_man = lambda vec_1, vec_2: np.linalg.norm(vec_1 - vec_2, axis=0)

    def to_np_arry(self):
        for item in self.path_info:
            item['np'] = np.array(item['traj'])

    def get_time_t(self, positionA):
        res = set()
        for item in self.path_info:
            ctr = 0
            for move in item["np"]:
                if np.array_equal(move, positionA):
                    res.add(ctr)
                ctr += 1
        print(list(res))

    def closet_path(self, pos_D):
        min_val = 100000
        for item in self.path_info:
            for pos_speed in item['np']:
                tmp_val = self.distance_man(pos_speed[0], pos_D)
                if min_val > tmp_val:
                    min_val = tmp_val
        return min_val


class QTable(object):

    def __init__(self, csv_table, csv_map, attcker_p, dico_game_setting):
        self.bins=2
        self.ctr=0
        self.regressor = RegressionFeature(attcker_p, dico_game_setting)
        self.action_d = {}
        self.matrix_f = None
        self.make_action_dict()
        self.df_raw = None
        self.map_df = None
        self.state_vector=None
        self.loader(csv_table, csv_map)

    def make_target_bins_nominal(self,bin=2):
        if bin==0:
            return
        assert(bin>1)
        bins=[0.0,1.0]
        counts = Counter(self.matrix_f[:,-1])
        d_remain = {key: counts[key] for key in counts if key not in bins}
        while bin>2:
            max_key = max(d_remain, key=lambda k: d_remain[k])
            bins.append(max_key)
            del d_remain[max_key]
            bin=bin-1
        y_item = sorted(list(counts.keys()))
        bins = sorted(bins)
        print("->",y_item,"\t",bins)
        bin_map = np.digitize(y_item,sorted(bins),right=True)
        print(bin_map)
        y_new = np.array([bins[bin_map[y_item.index(i)]] for i in self.matrix_f[:,-1]])
        self.matrix_f[:,-1]=y_new
        # print(y_new)
        # for i,item_i in enumerate(self.matrix_f[:,-1]):
        #     print("{}-->{}".format(item_i,y_new[i]))
        #     assert(bin_map[y_item.index(item_i)]==y_new[i])

    def loader(self, csv_table, csv_map):
        self.df_raw = pd.read_csv(csv_table, sep=';')
        self.map_df = pd.read_csv(csv_map, sep=';', names=["id", "state"])

        print(list(self.df_raw))

        self.make_features_df()
        self.make_target_bins_nominal()

    def make_features_df(self):
        self.state_vector = self.state_str_to_vec(self.df_raw['id'])
        print (self.state_vector.shape)
        state_matrix = self.state_vector.reshape(self.state_vector.shape[0],self.state_vector.shape[1]*self.state_vector.shape[2])
        matrix_val = np.asmatrix(self.df_raw[list(self.df_raw)[1:]].values)
        print("matrix_val.shape: ",matrix_val.shape)

        out_matrix = self.make_matrix_flat(matrix_val,state_matrix)
        self.matrix_f = out_matrix

    def make_matrix_flat(self,matrix,state_matrix):
        flat_matrix = np.array(matrix).flatten()
        print("flat_matrix.shape: ", flat_matrix.shape)
        a = np.empty((flat_matrix.shape[0], state_matrix.shape[1] + 4))
        print("a.shape: ", a.shape)
        action_number = len(list(self.df_raw)) - 1
        i = 0

        while i < matrix.shape[0]:
            for action_i in range(action_number):
                action_arr = self.action_d[action_i]
                idx = (i * action_number) + action_i
                a[idx, :-4] = state_matrix[i]
                a[idx, -4:-1] = action_arr
                a[idx, -1] = flat_matrix[idx]
            i = i + 1
        #np.savetxt("/home/ERANHER/car_model/generalization/foo.csv", a, delimiter=",")
        return a

    def make_matrix(self,matrix,state_matrix):
        print(matrix.shape)
        print(state_matrix.shape)
        a = np.concatenate((state_matrix, matrix), axis=1)
        print(a[0:2])
        return a



    def state_str_to_vec(self,state_id):
        return self.get_state_by_id(state_id)

    def func2(self,state_np,action_val,action_id):
        res = self.regressor.get_F(self.action_d[action_id], state_np, action_val)
        self.d_l.append(res)


    def get_state_by_id(self, id_num):
        str_state = self.map_df.loc[self.map_df['id'] == id_num, 'state']
        return np.array([QTable.state_string_state_object(xi) for xi in str_state])
        # (len(str_state) == 1)
        #return QTable.state_string_state_object(str_state)

    @staticmethod
    def state_string_state_object(str_state):
        arr = str(str_state).split("_")  # 0A_(18, 14, 1)_(2, 2, -1)_0|0D_(12, 12, 3)_(1, 1, 1)_1|
        A_pos = np.array(eval(arr[1]))
        A_speed = np.array(eval(arr[2]))
        D_pos = np.array(eval(arr[4]))
        D_speed = np.array(eval(arr[5]))
        state_np = np.array([A_pos, A_speed, D_pos, D_speed])
        return state_np

    def make_action_dict(self):
        ctr = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    self.action_d[ctr] = np.array([x, y, z])
                    ctr = ctr + 1
        print(self.action_d)

    def get_data_set(self):
        class_label = self.matrix_f[:, -1]  # for last column
        dataset = self.matrix_f[:, :-1]  # for all but last column
        return dataset,class_label


class RegressionFeature(object):

    def __init__(self, AttackerPath_obj, game_setting):
        self.attcker_path = AttackerPath_obj
        self.game_info_dict = game_setting
        self.distance_eq = lambda vec_1, vec_2: np.sqrt(np.sum((vec_1 - vec_2) ** 2))
        self.distance_man = lambda vec_1, vec_2: np.linalg.norm(vec_1 - vec_2, axis=0)
        self.distance_function=self.distance_man
        self.d={}
    def get_F(self, np_action,np_state, val):
        self.d = {"target":val}
        self.fix_f(np_state,np_action),val
        return self.d
    def get_near_path(self, posD, posA):
        dist = self.attcker_path.closet_path(posD)
        time_t = self.attcker_path.get_time_t(positionA=posA)
        return dist, time_t

    def goal_F(self,pos_agnet):
        l=[]
        for goal_pos in self.game_info_dict['P_G']:
            l.append(self.distance_function(pos_agnet,goal_pos))
        return np.array(np.array(l))

    def A_D(self,pos_D,pos_A):
        return self.distance_function(pos_D,pos_A)

    def size_F(self,pos_D):
        return self.distance_function(pos_D,self.game_info_dict['grid_size'])

    def fix_f(self,np_state,action):
        tmp={'A_pos:':np_state[0],'A_speed:':np_state[1],
                'D_pos:':np_state[2],'D_speed':np_state[3],'action':action}
        self.d.update(tmp)



def MainLoader():

    SEED=2000
    np.random.seed(SEED)
    home = expanduser("~")
    dir_data = "{}/car_model/generalization/data".format(home)
    print(dir_data)
    Q_csv = "{}/Q.csv".format(dir_data)
    p_csv = "{}/p.csv".format(dir_data)
    map_csv = "{}/map.csv".format(dir_data)
    con_csv = "{}/con21.csv".format(dir_data)


    loader = Loader(dir_data)
    loader.load_p(p_csv)
    loader.load_game_setting(con_csv)
    dico_info_game = loader.get_config_id(0)
    attacker_paths = loader.get_path_object()
    print(dico_info_game)

    q= QTable(Q_csv, map_csv, attacker_paths, dico_info_game)
    x,y = q.get_data_set()
    print(len(x))
    #x=x[:10000]
    #y=y[:10000]
    arr = Counter(y)
    new_arr = [x/len(y) for x in arr.values()]
    print(arr)
    print(new_arr)

    return x,y


if __name__ == "__main__":
    MainLoader()