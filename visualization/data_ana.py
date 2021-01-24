import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
def get_action_dens(df):

    colz = [(str(x),str(x+25)) for x in range(27)]
    d={}
    for new,old in colz:
        d[old]=new
    df = df.rename(d, axis='columns')
    col = list(df)
    #df = df[df[col[-1]]>=1]
    df = df.iloc[:,-28:-1]



    df['s']=df.sum(axis=1)/27
    df['r']=df['s']==df['0']

    print(Counter(df['r'].values))
    df = df[df['r']==False]
    del df['r']
    del df['s']

    df['max']=df.max(axis=1)
    df['argmax'] = df.idxmax(axis=1)

    print((list(df)))
    df['argmax'].hist()
    plt.show()
    print(df.head)

if __name__ == '__main__':
    p="/home/eranhe/car_model/debug/all.csv"
    df = pd.read_csv(p)
    #print(df.head())
    get_action_dens(df)
    pass