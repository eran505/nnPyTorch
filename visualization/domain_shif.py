
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os_util as pt
from sys import exit
from collections import Counter
import os

def agg_df(root_dir):
    res = pt.walk_rec(root_dir,[],'.txt')
    l = list(map(lambda x : pd.read_csv(x,names=['subject','target','t','x','y','z']),res))
    return pd.concat(l)

def target_class_distribution(root_dir,col='subject'):
    df = agg_df(root_dir)
    print(list(df))
    print("Size : ",len(df))
    print(" #activities : {}".format(len(df[col].unique())))
    df[col].hist()
    print(Counter(df[col].values))
    plt.show()

def merge_files(root_dir):
    res = pt.walk_rec(root_dir,[],".txt")
    res = res[-3:]
    l = list(map(lambda x : pd.read_csv(x,names=['subject','class','t','x','y','z']),res))
    for df_i in l:
        print(list(df_i))
        df_i['x'].plot( kind='line')
        #df_i['x'].rolling(window=60).mean().plot()#style='k')
        #df_i['x'].rolling(window=10).std().plot(style='b')

    plt.show()
    exit(0)
if __name__ == "__main__":

    f_folder = "/{}/eran/DA/DATASET/wisdm/wisdm-dataset/raw/phone/accel/"
    #merge_files(f_folder)
    file_ex1="{}/y_train.txt".format(f_folder)
    target_class_distribution(f_folder)
    pass