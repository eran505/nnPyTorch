from os_util import walk_rec,mkdir_system
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from CSV.utils_csver import read_multi_csvs
from CSV.get_info import  get_info_path_gen
# "episodes";"Collision";"Wall";"Goal"
import os_util as pt
from CSV.tabular_csv import get_both_csvs
color_array = ['red', 'green', 'blue', 'orange', 'gray', "yellow", "brown", "purple", "m","pink"]

def path_to_config(name):
    arr_data = str(name).split("_")
    d = {}
    for item in arr_data:
        if item[0] == "S":
            d["seed"] = item[1:]
        elif item[0] == "I":
            d['id_exp'] = item[1:]
        elif item[0] == "M":
            d['mode'] = int(item[1:])
        elif item[0] == "O":
            d['option'] = int(item[1:])
        elif item[0] == "H":
            d['h'] = int(item[1:])
        elif item[0] == "A":
            d['a'] = int(item[1:])
    d['name'] = '_'.join(str(name).split('_')[:-1])
    return d


def coverage_at_episode(csv_file_path,d=None):
    df = pd.read_csv(csv_file_path,sep=';')
    if d is None:
        d={}
    d["max_col"] = df["Collision"].max()
    d["ep_1000"] = df.loc[df["Collision"]==1000,"episodes"].min()
    d["ep_900"] = df.loc[df["Collision"]>900,"episodes"].min()
    d["ep_950"] = df.loc[df["Collision"] > 950, "episodes"].min()
    return d


def func(p="/home/eranhe/car_model/exp/paths"):
    res = pt.walk_rec(p,[],"_Eval.csv",lv=-2)
    d_l=[]
    for item in res:
        dico = path_to_config(os.path.basename(item))
        dico = coverage_at_episode(item,dico)
        base_dir = os.path.basename(os.path.dirname(item))
        dico["base_dir"]=int(base_dir)
        d_l.append(dico)
    df = pd.DataFrame(d_l)
    df.to_csv("{}/info.csv".format(p))
    df_900 = pd.pivot_table(df,values="ep_900",index="mode",columns="base_dir",aggfunc="mean")
    df_950 = pd.pivot_table(df,values="ep_950",index="mode",columns="base_dir",aggfunc="mean")
    df_1000 = pd.pivot_table(df,values="ep_1000",index="mode",columns="base_dir",aggfunc="mean")
    df_900.to_csv("{}/ep_90.csv".format(p))
    df_1000.to_csv("{}/ep_100.csv".format(p))
    df_950.to_csv("{}/ep_95.csv".format(p))
    df_0 = df.loc[df["mode"]==2].sort_values(by=['seed'])
    df_2 = df.loc[df["mode"] == 0].sort_values(by=['seed'])
    print(len(df_2))
    print(len(df_0))
    plt.scatter(df_2['ep_1000'].values,df_0['ep_1000'].values)
    plt.show()
    exit()
    #plt.plot()
if __name__ == '__main__':
    pp="/home/eranhe/car_model/exp/paths2"
    pp="/home/eranhe/car_model/debug"

    # func(pp)