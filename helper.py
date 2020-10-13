from collections import Counter

import pandas as pd
import os_util as pt
import matplotlib.pyplot as plt


def log_file(list_items,to_path,column_names):
    df = pd.DataFrame(list_items, columns=column_names)
    df.to_csv(to_path,sep=',')

def plot_loss(path):
    father_p = '/'.join(str(path).split('/')[:-1])
    df = pd.read_csv(path,index_col=0)
    print(list(df))
    df["loss"].plot( kind='line')
    plt.savefig('{}/plot.png'.format(father_p))  # save the figure to file
    plt.show()

def value_statistic(matrix_df):
    size = len(matrix_df[:,-27:].flatten())
    c = Counter(matrix_df[:,-27:].flatten())
    xx= [(i, c[i] / size ) for i in c]
    sorted_by_second = sorted(xx, key=lambda tup: tup[1])
    reversed(sorted_by_second)
    for item in sorted_by_second:
        print("{},{}".format(item[0],item[1]))
    exit()

def concat_df():
    res = pt.walk_rec("/home/ERANHER/car_model/generalization/4data/dataNN",[],".csv")
    l=[]
    for item in res:
        l.append(pd.read_csv(item))
    df_all = pd.concat(l)
    df_all.to_csv("/home/ERANHER/car_model/generalization/4data/dataNN/all.csv",index=False)



if __name__ == "__main__":
    concat_df()
    exit()
    plot_loss("/home/ERANHER/car_model/nn/loss_train.csv")
    pass