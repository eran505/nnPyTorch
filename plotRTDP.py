import numpy as np
import os_util as pt
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import groupby

def get_all_df_sort_by_size(p_path="/home/eranhe/car_model/debug"):
    res = pt.walk_rec(p_path, [], ".csv")
    d_path = {}
    d_size = {}
    for item_csv in res:
        ky = str(item_csv).split('/')[-1].split('.')[0]
        if ky=="d_states":
            continue
        size_i = os.path.getsize(item_csv)
        d_path[ky] = item_csv
        d_size[ky] = size_i
    print(d_path)
    print(d_size)
    sort_tuples = sorted(d_size.items(), key=lambda x: x[1], reverse=True)
    sort_tuples_all = []
    for k, v in sort_tuples:
        sort_tuples_all.append((k, v, d_path[k]))
    print(sort_tuples_all)
    return sort_tuples_all
def filter_top_k(k,df):
    d={}
    for i in list(range(0,27)):
        d[str(i)]=df[str(i)].iloc[-1]
    sort_tuples = sorted(d.items(), key=lambda x: x[1], reverse=True)
    sort_tuples_cut = sort_tuples[:k]
    res = [k for k,v in sort_tuples_cut]
    return res

def plot_graph(list_sorted_tuple):
    list_col_str = [str(x) for x in list(range(0, 27) )]
    for key,size,path in list_sorted_tuple:
        print("{} {} {}".format("-"*10,key,"-"*10))
        df = pd.read_csv(path, names=list_col_str)
        list_col=list(df)
        list_col = filter_top_k(10,df)
        df[list_col].plot()
        plt.show()

def all_policy_eval(path_p,col_name="ctr_coll"): ##"ctr_round","ctr_wall","ctr_coll","ctr_at_goal"
    res = pt.walk_rec(path_p, [], "P.csv")
    d = {}
    df_list = []
    newDF = pd.DataFrame()
    for item in res:
        name = "_".join(str(item).split('/')[-1].split('.')[0].split('_')[:-1])
        df_i = pd.read_csv(item)
        df_list.append(df_i)
        newDF["{}-{}".format(name,col_name)]=df_i[col_name]/500

    newDF.plot(kind='line')
    plt.savefig('{}/plot.png'.format(path_p))  # save the figure to file
    plt.show()


def merge_all_policy_eval(path_p):
    res = pt.walk_rec(path_p, [], "P.csv")
    d={}
    df_list=[]
    for item in res:
        df_list.append(pd.read_csv(item))
    df_all =pd.concat(df_list).groupby(level=0).mean()
    #dfObj = df_all.transpose()
    print(list(df_all))
    l_col=['ctr_wall', 'ctr_coll', 'ctr_at_goal','ctr_open']
    l_col_ratio=[("_".join(str(x).split('_')[1:])+"_ratio").title() for x in l_col ]
    for col,col_new in zip(l_col,l_col_ratio):
        df_all[col_new] = df_all[col] / 500
    df_all= df_all.set_index('ctr_round')
    df_all[l_col_ratio].plot( kind='line')
    plt.savefig('{}/plot.png'.format(path_p))  # save the figure to file
    plt.show()


def coundEndPoint(dir_path,filter_key='_8'):
    res = pt.walk_rec(p, [], ".csv")
    for item in res:
        if str(item).split('/')[-1].__contains__(filter_key) is False:
            continue


def trajectory_read(p_path):
    df = pd.read_csv(p_path, names=['tran'])
    print(list(p_path))
    idx = (df['tran'] == "END").idxmax()
    print(idx)
    df = df.iloc[idx + 1:]
    array = df['tran'].to_numpy()
    res = [list(g) for k, g in groupby(array, lambda x: x != "END") if k]
    return res



if __name__ == "__main__":
    p="/home/ERANHER/car_model/results/26_04/con1"
    p_path="/home/ERANHER/car_model/results/dataEXP/old/sizeExp/roni/out"
    p_path="/home/ERANHER/car_model/exp"
    all_policy_eval(p_path,"ctr_at_goal") #ctr_at_goal  ctr_open
    merge_all_policy_eval(p_path)
    #merge_all_policy_eval()
    #res = merge_all_policy_eval(p)
