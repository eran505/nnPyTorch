import numpy as np
import os_util as pt
import pandas as pd
import os
import matplotlib.pyplot as plt
from os.path import expanduser
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

def read_multi_csvs(p):
    big_l=[]
    l_csv = []
    with open(p, 'r') as f:
        for line in f:
            line = str(line).replace(" ","").replace('\n','').replace('""','')
            if len(line)<1:
                continue
            d={}
            if(str(line).startswith("\"e")):
                big_l.append(l_csv)
                l_csv=[]
                cols = str(line).split(';')
                continue
            line_arr = str(line).split(";")
            for i,col in enumerate(cols):
                d[col]=float(line_arr[i])
            l_csv.append(d)
    if(len(l_csv)>0):
        big_l.append(l_csv)
    df_l=[]
    for item in big_l:
        if len(item)>0:
            df_l.append(pd.DataFrame(item))

    return df_l

def one_path_ana(path_p="/home/eranhe/car_model/one_path"):
    print(path_p)
    l=[]
    res = pt.walk_rec(path_p,[],"Eval.csv")
    for item_csv_p in res:
        d = {}
        print(item_csv_p)
        d["seed_number"] = str(item_csv_p).split("/")[-1].split("_")[0]
        d["u_id"] = str(item_csv_p).split("/")[-1].split("_")[1]
        d["max_L"] = str(item_csv_p).split("/")[-1].split("_")[2]
        d["df_l"] = read_multi_csvs(item_csv_p)

        df_pre = mean_multi_dfs(d["df_l"][:-1])
        last_row = d["df_l"][-1].tail(1).mean(axis=0).to_dict()
        for ky in last_row:
            d["{}".format(ky).replace('\"','')] =last_row[ky]
        d['path_size'] = len(d["df_l"]) - 1
        for ky in df_pre:
            d["{}_{}".format("P",ky).replace('\"','')]=df_pre[ky]
        del d["df_l"]
        l.append(d)
    df = pd.DataFrame(l)
    df.to_csv("{}/all.csv".format(path_p))
    return df
def mean_multi_dfs(l_df,tail_row=1):
    means_arr=[]
    for df in l_df:
        tail_df = df.tail(tail_row)
        mean = tail_df.mean(axis=0)
        means_arr.append(mean)
    df_all = pd.DataFrame(means_arr)
    df_mean = df_all.mean(axis=0)
    return df_mean.to_dict()


def one_path(path_p="/home/eranhe/car_model/out"):
    print(path_p)
    l = []
    res = pt.walk_rec(path_p, [], "Eval.csv")
    for item_csv_p in res:
        d = {}
        print(item_csv_p)
        d["seed_number"] = str(item_csv_p).split("/")[-1].split("_")[0]
        d["u_id"] = str(item_csv_p).split("/")[-1].split("_")[1]
        d["max_L"] = str(item_csv_p).split("/")[-1].split("_")[2]
        df = pd.read_csv(item_csv_p,sep=';')
        df = df.dropna()
        df_tail = df.tail(1)
        last_row = df_tail.mean(axis=0).to_dict()
        for ky in last_row:
            d["{}".format(ky).replace('\"','')] =last_row[ky]
        l.append(d)
    df = pd.DataFrame(l)
    df.to_csv("{}/all_one_path.csv".format(path_p))
    return df

if __name__ == "__main__":

    home = expanduser("~")
    df1 = one_path_ana("{}/car_model/one_path".format(home))
    df2 = one_path("{}/car_model/out".format(home))
    print(list(df1))
    print(list(df2))
    df3 = df1.append(df2)
    df3.to_csv("{}/car_model/df.csv".format(home))

    exit()
    p="/home/ERANHER/car_model/results/26_04/con1"
    p_path="/home/ERANHER/car_model/results/dataEXP/old/sizeExp/roni/out"
    p_path="/home/ERANHER/car_model/exp"
    all_policy_eval(p_path,"ctr_at_goal") #ctr_at_goal  ctr_open
    merge_all_policy_eval(p_path)
    #merge_all_policy_eval()
    #res = merge_all_policy_eval(p)
