import pandas as pd
from os.path import expanduser
import numpy as np
import hashlib

class Qpolicy(object):

    def __init__(self):
        self.home = expanduser("~")
        self.Q=None
        self.rel=None
        self.map= {}
        self.matrix_f=None
        self.loader_all_data()

    def loader_all_data(self,csv_table="/home/ERANHER/car_model/debug/Q.csv",csv_map="/home/ERANHER/car_model/debug/map.csv"):
        names = ["S" + str(i) for i in range(1, 13)]
        names.insert(0, "id")

        df_raw = pd.read_csv(csv_table, sep=';')
        map_df = pd.read_csv(csv_map, sep=';', names=names)
        map_df['id'] = map_df['id'].astype('uint64')

        df_raw = pd.merge(map_df, df_raw, how='inner', on=['id'])
        ctr_df = self.get_count_state()
        df_raw = pd.merge(df_raw, ctr_df, how='left', on=['id'])
        df_raw['ctr'].fillna(1,inplace=True)
        self.colz=list(df_raw)
        print(self.colz)
        df_raw.to_csv("{}/car_model/generalization/5data/f.csv".format(self.home),index=False)
        self.matrix_f = df_raw.to_numpy()
        idz = self.matrix_f[:,0]
        for i,id_number in enumerate(idz):
            self.map[self.hash_func(self.matrix_f[i,1:13])]=(i,id_number)
        print("END")

    def hash_func(self,dataPoint):
        return hash(tuple(dataPoint))

    def get_count_state(self,):
        colz = ["S" + str(i) for i in range(1, 13)]
        colz.insert(0, "id")
        colz.append("ctr")

        df_last_states = pd.read_csv("{}/car_model/debug/Last_States.csv".format(self.home),names=colz,sep=';')
        df_last_states['id'] = df_last_states['id'].astype('uint64')

        return df_last_states[["id","ctr"]]


if __name__ == "__main__":
    q = Qpolicy()
    pass