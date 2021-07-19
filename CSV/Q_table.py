import pandas as pd
import os
import helper as hlp
from copy import deepcopy


def diff_tuple(a, b):
    l = []
    for i in range(len(a)):
        l.append(abs(a[i] - b[i]))
    return l


def add_speed(a):
    l = []
    for x in a:
        if x > 1:
            x = 1
        elif x < -1:
            x = -1
        l.append(x)
    return l


def get_grid_size():
    df = pd.read_csv("/home/eranhe/car_model/debug/con2.csv")
    dz = df.to_dict()
    x = dz["X"][0]
    y = dz["Y"][0]
    z = dz["Z"][0]
    h = dz["h"][0]
    return [x, y, z], h


CollReward = 1
GoalReward = -0.9
WallReward = -1
Step_reward = 0.0
discountF = 0.987


class State(object):
    loc_p = None
    speed_p = None
    loc_e = None
    speed_e = None
    time_t = -1
    jumps = -1
    belief_states = []

    def to_list(self):
        return [self.time_t, self.loc_e, self.speed_e, self.loc_p, self.speed_p, self.belief_states, self.jumps]


def string_to_state(string_state):
    # print(string_state)
    if len(string_state) < 10:
        return None
    s = State()
    s.time_t = int(string_state[0])
    s.jumps = int(string_state[-1])
    arr = str(string_state).split('_')
    s.loc_e = eval(arr[2])
    s.speed_e = eval(arr[3][:-2])
    s.loc_p = eval(arr[4])
    s.speed_p = eval(arr[5][:-1])
    new_arr = arr[6].replace("{", "").replace("}", "")
    l = []
    for i in new_arr:
        i = i.replace(" ", "")
        if len(i) == 0:
            continue
        l.append(i)
    new_arr = [int(x) for x in l]
    s.belief_states = new_arr
    return s.to_list()


def hurstic(state_s, all_paths):
    state_s = eval(state_s)
    vec = []
    old_state = deepcopy(state_s)
    # print("in_state=", old_state)
    for i in range(len(d_actions)):
        action = d_actions[i]
        state_s[4] = [sum(x) for x in zip(list(action), list(state_s[4]))]
        state_s[4] = add_speed(state_s[4])
        state_s[3] = [sum(x) for x in zip(list(state_s[3]), list(state_s[4]))]
        if int(h_mode) == 0:
            v = h0(state_s, all_paths)
        if int(h_mode) == 2:
            v = h2(state_s, all_paths)
        if int(h_mode) == 3:
            v = h3(state_s, all_paths)
        if int(h_mode) == 4:
            v = h4(state_s, all_paths)
        if int(h_mode) == 5:
            v = h5(state_s, all_paths)
        # print(action, "\t", state_s, "  = ", v)
        vec.append(v)
        state_s = deepcopy(old_state)
    return vec


def check_out_of_bound(state):
    if state[3][0] >= grid[0] or state[3][0] < 0:
        return 1
    if state[3][1] >= grid[1] or state[3][1] < 0:
        return 1
    if state[3][2] >= grid[2] or state[3][2] < 0:
        return 1
    return 0


def h0(s, all_paths):
    if check_out_of_bound(s) == 1:
        return WallReward * pow(discountF, s[-1])
    return CollReward * pow(discountF, int(s[0]))


def h2(s, all_paths):
    if check_out_of_bound(s) == 1:
        return WallReward * pow(discountF, s[-1])
    l = []
    for b in range(len(all_paths)):
        pathz = all_paths[b]['traj']
        for idx, item in enumerate(pathz[:-1]):
            locE = item[0]
            max_d = max(diff_tuple(locE, s[3]))
            l.append(max_d)
    min_d = min(l)
    return CollReward * pow(discountF, min_d)


def h3(s, all_paths):
    if check_out_of_bound(s) == 1:
        return WallReward * pow(discountF, s[-1])
    l = []
    for b in s[-2]:
        pathz = all_paths[b]['traj']
        for idx, item in enumerate(pathz[:-1]):
            locE = item[0]
            max_d = max(diff_tuple(locE, s[3]))
            l.append(max_d)
    min_d = min(l)
    time = int(s[0]) + int(s[-1])
    return CollReward * pow(discountF, min_d + time)

def h4(s, all_paths):
    if check_out_of_bound(s) == 1:
        return WallReward * pow(discountF, s[-1])
    l = []
    for b in s[-2]:
        pathz = all_paths[b]['traj']
        for idx, item in enumerate(pathz[:-1]):
            locE = item[0]
            max_d = max(diff_tuple(locE, s[3]))
            x = max(max_d,idx)
            l.append(x)
    min_d = min(l)
    time = int(s[0]) + int(s[-1])
    return CollReward * pow(discountF, min_d + time)

def h5(s, all_paths):
    if check_out_of_bound(s) == 1:
        return WallReward * pow(discountF, s[-1])
    for b in s[-2]:
        l = [len(b)]
        pathz = all_paths[b]['traj']
        for idx, item in enumerate(pathz[:-1]):
            locE = item[0]
            max_d = max(diff_tuple(locE, s[3]))
            if idx<=max_d:
                x = max(max_d,idx)
                l.append(x)
    min_d = min(l)
    time = int(s[0]) + int(s[-1])
    return CollReward * pow(discountF, min_d + time)




def make_action_D():
    l = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                l.append((x, y, z))
    return l


def read_state_map():
    df_map = pd.read_csv("/home/eranhe/car_model/debug/MAP.csv", sep=';')
    print(list(df_map))
    df_map['State'].dropna(inplace=True)
    df_map['State'] = df_map['State'].apply(lambda x: string_to_state(x))
    return df_map


def start_f(df_map, df_Q, all_paths):
    n = 100
    ep = 1.97e-06
    sum = 0
    df_Q_sample = df_Q.sample(n)
    d = df_Q_sample.to_dict('records')
    for item in d:
        key = item['ID']
        state = df_map.loc[df_map['ID'] == key, "State"]
        print("----", key, "----")
        if len(state) != 1:
            print(key, ":", len(state))
            continue
        state = state.iloc[0]
        h_vec = hurstic(state, all_paths)
        max_diff = 0
        for i, val in enumerate(h_vec):
            if max_diff < val - item[str(i)]:
                max_diff = val - item[str(i)]
            if item[str(i)] > val and ep < item[str(i)] - val:
                print(i, " : ", key, "\t diff:", item[str(i)] - val)
        sum += max_diff
        print("MAX:", max_diff)

    print("mean: ", sum / n)


def main():
    if os.path.isfile("/home/eranhe/car_model/debug/map_object.csv") is False:
        df_map = read_state_map()
        df_map.to_csv("/home/eranhe/car_model/debug/map_object.csv", sep=';', index=False)
    df_map = pd.read_csv("/home/eranhe/car_model/debug/map_object.csv", sep=';')
    df_Q = pd.read_csv("/home/eranhe/car_model/debug/Q.csv", skiprows=1, sep=';')
    return df_Q, df_map


d_actions = make_action_D()
grid, h_mode = get_grid_size()

if __name__ == '__main__':
    l = hlp.load__p("/home/eranhe/car_model/debug/p.csv")
    df_Q, df_map = main()
    start_f(df_map, df_Q, l)
