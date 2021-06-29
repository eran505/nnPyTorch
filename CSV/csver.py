
from os_util import walk_rec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from CSV.utils_csver import read_multi_csvs
# "episodes";"Collision";"Wall";"Goal"
def namer(mode,option,h):
    name=""
    if mode == 0:
        name+="RTDP(position,"
    elif mode ==1:
        name+="RTDP(plan,"
    elif mode ==2:
        name+="RTDP(posT,"

    if option==0:
        name+=" atomic,"
    elif option==1:
        name += " option,"

    if h == 0:
        name += " zero"
    elif h == 1:
        name += " all_path_next_step"
    elif h == 2:
        name += " rel_path_next_step"
    elif h == 3:
        name += "all_paths_end"
    elif h == 4:
        name += "all_paths_end_fast"


    name+=")"
    return name

def con_to_dico(path_to_con="/home/eranhe/car_model/debug/con16.csv"):
    x=5
    father = os.path.dirname(path_to_con)
    d_data,max_ep = plot_all_csvs()
    df = pd.read_csv(path_to_con)
    df['MAX_Collision']=df.apply(Max_coll,data=d_data,axis=1)
    df['AVG_Collision_{}'.format(x)]=df.apply(avg_coll,data=d_data,tail=x,axis=1)
    df['episodes']=df.apply(get_T,data=d_data,axis=1)
    df['MAX_episodes']=max_ep
    for item_ky in d_data.keys():
        df_data = d_data[item_ky]['df']
        if d_data[item_ky]['mode']==-1:
            val =  d_data[item_ky]['df']['Collision'][-x:].mean()
            arr = np.full(int(max_ep/10), val)
            plt.plot(arr, label="Plan_Rec",color='k')
        else:
            name = namer(d_data[item_ky]['mode'],d_data[item_ky]['option'],d_data[item_ky]['h'])
            plt.plot(df_data['Collision'][:int(max_ep/10)].values , label= name,ls='--' )

    print(df[["ID","X","Y","Z","h","o","m","MAX_Collision",'AVG_Collision_{}'.format(x),"episodes","MAX_episodes"]])
    df.to_csv("{}/res.csv".format(father))
    plt.legend()
    plt.savefig("{}/plot.png".format(father))
    plt.show()

def Max_coll(row,data):
    id_ = str(row['ID'])
    data = data[id_]
    return data['df']['Collision'].max()

def avg_coll(row,data,tail):
    id_ = str(row['ID'])
    data = data[id_]
    res = data['df']['Collision'][-tail:].mean()
    return res

def get_T(row,data):
    id_ = str(row['ID'])
    data = data[id_]
    res = len(data['df']['Collision'].values)
    return res


def path_to_config(name):
    arr_data = str(name).split("_")
    d={}
    for item in arr_data:
        if item[0]=="S":
            d["seed"]=item[1:]
        elif item[0]=="I":
            d['id_exp']=item[1:]
        elif item[0]=="M":
            d['mode']=int(item[1:])
        elif item[0]=="O":
            d['option']=int(item[1:])
        elif item[0]=="H":
            d['h']=int(item[1:])
        elif item[0]=="A":
            d['a']=int(item[1:])
    d['name']='_'.join(str(name).split('_')[:-1])
    return d

def plot_all_csvs(dir_path="/home/eranhe/car_model/debug"):
    res = walk_rec(dir_path,[],"Eval.csv")
    d_l={}
    max_ep=0
    for item in res:
        name = str(item).split('/')[-1].split('.')[0]
        d = path_to_config(name)
        df=pd.read_csv(item,sep=';')
        df.dropna(axis=0,inplace=True)
        d['df'] = df
        if max_ep<len(df):
            max_ep=len(df)
        d_l[d["id_exp"]]=d
    return d_l,max_ep

def make_collision_prop(df_l,name):
    # "episodes";"Collision";"Wall";"Goal"
    size = len(df_l)
    if size==1:
        return
    start_from=[]
    for df_i in df_l[:-1]:
        df_i['"Collision"']=df_i['"Collision"']/(size-1)
        start_from.append(df_i['"Collision"'])
    df_subs = pd.concat(start_from)
    y_arr = np.arange(0, len(df_subs), 1)
    plt.plot(y_arr, df_subs.values, label=name, ls=':')
    y_arr = np.arange(len(df_subs), len(df_subs) + len(df_l[-1]['"Collision"'].values), 1)
    #plt.plot(y_arr, df_l[-1]['"Collision"'].values, label=name + "_F", ls='--')
    plt.legend()
    plt.show()

def tmp(list_csvs):
    d={}
    thershold=0
    for item in list_csvs:

        learn_id = str(item).split("/")[-1].split("_")[2]

        l_df = read_multi_csvs(item)

        # if l_df[-1]['"Collision"'].iloc[-1]!=1.0:
        #     continue

        for df_i in l_df[:-1]:
            df_i['"Collision"'] = df_i['"Collision"'] / (len(l_df)-1)
        df = pd.concat(l_df)
        #print(list(df))
        #l_coll.append(df['"Collision"'].values)
        if learn_id not in d:
            d[learn_id]=[]
        d[learn_id].append(df['"Collision"'].values)

    for ky in d:

        l_coll=d[ky]
        max_line = 0


        b = np.full([len(l_coll), len(max(l_coll, key=lambda x: len(x)))], 0,dtype=float)
        for i, j_arr in enumerate(l_coll):
            b[i,:len(j_arr)] = j_arr
            if len(j_arr)<b.shape[1]:
                b[i,len(j_arr):] = max(j_arr)
        b[b<thershold]=0


        plt.plot(b.mean(axis=0)[:], ls='--')
    # print(ky_uid,"<----")
    plt.legend()
    # save_dir = pt.mkdir_system("{}/car_model/figs".format(expanduser('~')),str(list_csvs[0]).split('/')[-2],False)
    # plt.savefig("{}/u{}_th={}_fig.png".format(save_dir,ky_uid,thershold))
    plt.show()
    print("end")

def just_plot(dir_path="/home/eranhe/car_model/debug"):
    res = walk_rec(dir_path,[],"Eval.csv")
    dico_list=[]
    for item in res:
        name = str(item).split('/')[-1].split('.')[0]
        d = path_to_config(name)
        df_list=read_multi_csvs(item)
        for x in df_list:
            x.dropna(axis=0, inplace=True)
        d['df'] = df_list
        dico_list.append(d)
    return dico_list

if __name__ == '__main__':
    tmp(["/home/eranhe/car_model/debug/S3_I9_M1_O1_H1_A1_Eval.csv"])
    exit()
    dico_d = just_plot()
    for item in dico_d:
        make_collision_prop(item['df'],item['id_exp'])

    #con_to_dico()