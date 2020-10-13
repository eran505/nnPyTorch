

import pandas as pd
import os_util as pt
import matplotlib.pyplot as plt


def log_file(list_items,to_path,column_names):
    df = pd.DataFrame(list_items, columns=column_names)
    df.to_csv(to_path,sep=',')

def plot_loss(path):
    df = pd.read_csv(path,index_col=0)
    print(list(df))
    df["loss"].plot( kind='line')
    #plt.savefig('{}/plot.png'.format(path_p))  # save the figure to file
    plt.show()





if __name__ == "__main__":
    plot_loss("/home/ERANHER/car_model/nn/loss_train.csv")
    pass