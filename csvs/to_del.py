from os_util import walk_rec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# "episodes";"Collision";"Wall";"Goal"

def plot_all_csvs(dir_path="/home/ERANHER/car_model/debug"):
    res = walk_rec(dir_path,[],"Eval.csv")
    print(res)
    for item in res:
        name = str(item).split('/')[-1].split('.')[0]
        mode = int(str(name).split("_")[2][-1])
        df_i = pd.read_csv(item,sep=';')
        df_i.dropna(inplace=True)
        # print(df_i['Collision'].values)
        plt.plot(df_i['Collision'].values,label=name)
        # print(np.mean(df_i['Collision'].values))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_all_csvs()