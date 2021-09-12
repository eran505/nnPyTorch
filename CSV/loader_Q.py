
import pandas as pd
import os
import helper as hlp
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
HOME = os.path.expanduser("~")

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

    arr = str(string_state).split('_')
    s.time_t = int(arr[0])
    s.jumps = int(arr[-1])
    s.loc_e = eval(arr[2])
    s.speed_e = eval(arr[3][:-2])
    s.loc_p = eval(arr[4])
    s.speed_p = eval(arr[5][:-1])
    str_a=""
    b=[]
    for i in arr[6]:
        if i=="]":
            str_a+="]"
            b.append(eval(str_a))
            str_a=""
        else:
            str_a+=i
    s.belief_states=b
    return s.to_list()


def read_state_map():
    df_map = pd.read_csv("{}/car_model/debug/all_MAP.csv".format(HOME), sep=';')
    print(list(df_map))
    df_map['State'].dropna(inplace=True)
    df_map['State'] = df_map['State'].apply(lambda x: string_to_state(x))
    return df_map

def make_action_D():
    l = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                l.append((x, y, z))
    return l

def get_df_features(df,size_paths):
    rec = df.to_dict('records')
    a = np.zeros(shape=(len(df), size_paths+13+1+27))
    col = ["f_"+str(x) for x in range(size_paths+13)]
    a_col = ["a_"+str(x) for x in range(27)]
    col.insert(0,'ID')
    col=col+a_col
    for i,item in enumerate(rec):
        state_str = eval(item['State'])
        a[i,0]=item['ID']
        a[i,1]=state_str[0]
        a[i][2:5] = state_str[1]
        a[i][5:8] = state_str[2]
        a[i][8:11] = state_str[3]
        a[i][11:14] = state_str[4]
        for x in state_str[5]:
            for x_i in x:
                a[i][14+x_i] = 1

        for j in range(27):
            a[i][14+size_paths+j]=item[str(j)]
    df_f = pd.DataFrame(a,columns=col)
    df_f['ID'] = df_f['ID'].astype('uint64')
    return df_f





def laoder_main():
    l = hlp.load__p("{}/car_model/debug/p.csv".format(HOME))
    if os.path.isfile("{}/car_model/debug/map_object.csv".format(HOME)) is False:
        df_map = read_state_map()
        df_map.to_csv("{}/car_model/debug/map_object.csv".format(HOME), sep=';', index=False)
    df_map = pd.read_csv("{}/car_model/debug/map_object.csv".format(HOME), sep=';')
    df_Q = pd.read_csv("{}/car_model/debug/all_Q.csv".format(HOME), sep=';')
    df_big = pd.merge(df_map,df_Q, how='inner', on=['ID'])

    df_f = get_df_features(df_big,len(l))

    


if __name__ == '__main__':
    laoder_main()