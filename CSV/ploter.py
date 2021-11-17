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

# def change_files():
#     new_id=2
#     p="/home/eranhe/car_model/exp/matrix/3"
#     res = pt.walk_rec(p,[],"_Eval.csv",lv=-1)
#     for item in res:
#         name = str(item).split(os.sep)[-1][:-4]
#         id_str = name.split("_")[1][:2]
#         name_new = name.replace(id_str,"I2")
#         command_mv = "mv {} {}{}{}{}".format(item,p,os.sep,name_new,".csv")
#         print(command_mv)
#         os.system(command_mv)
#     exit()

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
    d["max_iter"] = df["episodes"].max()
    d["max_col"] = df["Collision"].max()
    d["tail"] = df["Collision"].values[-1]
    d["ep_max"] =  df.loc[df["Collision"]>=d["max_col"]-10,"episodes"].min()
    d["state_coll"] = df.loc[df["Collision"] >=  d["max_col"], "States"].min()
    d["state_max"]=df["States"].max()
    list_ep=[x*100 for x in range(1,11)]
    for item in list_ep:
        d["ep_{}".format(item)] = df.loc[df["Collision"]>=item,"episodes"].min()
    return d

di={-1:"Baseline",0:"Position",1:"Belief",2:"Position_t"}

def func(p="/home/eranhe/car_model/exp/paths"):
    res = pt.walk_rec(p,[],"_Eval.csv",lv=-2)
    d_l=[]
    dir_to = mkdir_system(p,"res")
    for item in res:
        dico = path_to_config(os.path.basename(item))
        dico = coverage_at_episode(item,dico)
        base_dir = os.path.basename(os.path.dirname(item))
        dico["base_dir"]=int(base_dir)
        d_l.append(dico)
    df = pd.DataFrame(d_l)
    df.replace({"mode": di},inplace=True)
    df.to_csv("{}/info.csv".format(p))
    colz = list(filter(lambda x : str(x).__contains__('ep_'),list(df)))
    colz.extend(["max_col","tail","state_max","state_coll"])
    MAX_T =  df['max_iter'].max()
    print("MAX EP:", df['max_iter'].max())
    for col in colz:
        df_i = pd.pivot_table(df,values=col,index="mode",columns="base_dir",aggfunc="mean")
        if col.__contains__('ep'):
            df_i=round(df_i/MAX_T,2)
        if col.__contains__('col'):
            df_i =round(df_i / 1000,2)

        df_i.to_csv("{}/{}.csv".format(dir_to,col))
    exit()

if __name__ == '__main__':
    pp="/home/eranhe/car_model/exp/cc"
    pp="/home/eranhe/car_model/debug"
    pp="/home/eranhe/car_model/exp/new/big_gap"
    pp="/home/eranhe/car_model/exp/matrix"
    func(pp)