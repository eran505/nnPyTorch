from os_util import mkdir_system,walk_rec
import pandas as pd
from os import path
import os
import numpy as np

def rearrange_episodes(l_df,col="episodes"):
    for df in l_df:
        size = len(df)
        df[col]=np.arange(1000,1000*(size+1),1000)
        df.reset_index(drop=True, inplace=True)

    return l_df

def add_row_n(small_df,n_rows,num=1000,copy_last=False):
    if n_rows<1:
        return small_df
    if small_df.iloc[-1]["Collision"]==num or copy_last:
        small_df = small_df.append(small_df.iloc[[-1] * n_rows])
    else:
        avg = np.mean(small_df["Collision"].values[-3:])
        small_df.iloc[-1]["Collision"]=avg
        small_df = small_df.append(small_df.iloc[[-1] * n_rows])

    return small_df

def make_all_df_equal_len(l_df,is_coll=True):
    max_len = max(list(map(lambda x: len(x),l_df)))
    if is_coll:
        l_df_new = list(map(lambda x: add_row_n(x, abs(len(x) - max_len), 1000), l_df))
    else:
        l_df_new = list(map(lambda x: add_row_n(x, abs(len(x) - max_len), -1, copy_last=True), l_df))
    return l_df_new

def make_new_data(res,p,is_coll=True):
    df_list = list(map(lambda x: pd.read_csv(x,sep=',',index_col=0),res))
    names = list(map(lambda x:str(path.basename(x))[:-4] , res))
    df_list = make_all_df_equal_len(df_list)
    print(names)
    df_list = make_all_df_equal_len(df_list,is_coll)
    df_list = rearrange_episodes(df_list)
    len_max = len(df_list[0])
    # bin = len_max/cuts
    results = df_list[0][['episodes']].copy()
    results.reset_index(drop=True, inplace=True)
    results["percentage"] = results["episodes"]/results["episodes"].max()
    results["percentage"] = results["percentage"].round(3)
    for idx,df in enumerate(df_list):
        if is_coll:
            results[names[idx]] = df["Collision"]
        else:
            results[names[idx]] = df["States"]
    if is_coll:
        results.to_csv("{}/res_Coll.csv".format(p))
    else:
        results.to_csv("{}/res_Gen.csv".format(p))
    print("done")

def get_both_csvs(res,p):
    make_new_data(res, p, True)
    make_new_data(res, p, False)

def make_unnormalize_csv(dir_path,is_coll):
    if is_coll:
        name="res_Coll.csv"
    else:
        name = "res_Gen.csv"
    res = walk_rec(dir_path,[],name)
    l=[]
    for item in res:
        name_exp = str(item).split(os.sep)[-3]
        df = pd.read_csv(item,sep=',')
        df['exp']=name_exp
        l.append(df)
    df_big = pd.concat(l,axis=1)
    if is_coll:
        df_big.to_csv("{}/big_Coll.csv".format(dir_path))
    else:
        df_big.to_csv("{}/big_Gen.csv".format(dir_path))

if __name__ == '__main__':
    make_unnormalize_csv("/home/eranhe/car_model/exp/path2",False)
    make_unnormalize_csv("/home/eranhe/car_model/exp/path2",True)
    exit(0)
    is_coll=True
    p="/home/eranhe/car_model/debug/mean"
    res = walk_rec(p,[],"_Eval.csv",lv=-1)
    make_new_data(res,p,is_coll)
    make_new_data(res, p, False)