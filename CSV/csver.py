from os_util import walk_rec,mkdir_system
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from CSV.utils_csver import read_multi_csvs
from CSV.get_info import  get_info_path_gen
# "episodes";"Collision";"Wall";"Goal"
from CSV.tabular_csv import get_both_csvs
color_array = ['red', 'green', 'blue', 'orange', 'gray', "yellow", "brown", "purple", "m","pink"]


def namer(mode, option, h):
    name = ""
    if mode == 0:
        name += "RTDP(position,"
    elif mode == 1:
        name += "RTDP(plan,"
    elif mode == 2:
        name += "RTDP(posT,"

    if option == 0:
        name += " atomic,"
    elif option == 1:
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

    name += ")"
    return name


def con_to_dico(path_to_con="/home/eranhe/car_model/exp/2/con16.csv"):
    constant = 1000
    x = 5
    father = os.path.dirname(path_to_con)
    d_data, max_ep = plot_all_csvs(father)
    df = pd.read_csv(path_to_con)
    df['MAX_Collision'] = df.apply(Max_coll, data=d_data, axis=1)
    df['AVG_Collision_{}'.format(x)] = df.apply(avg_coll, data=d_data, tail=x, axis=1)
    df['episodes'] = df.apply(get_T, data=d_data, axis=1)
    df['MAX_episodes'] = max_ep
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for item_ky in d_data.keys():
        if (d_data[item_ky]['h'] == 3 or d_data[item_ky]['h'] == 0) is False:
            continue
        b = int(max_ep * 0.7)
        x_axis = np.array([1e3 * x for x in range(b)]).astype(np.long)
        df_data = d_data[item_ky]['df']
        df_data['Collision'] = df_data['Collision'] / constant
        if d_data[item_ky]['mode'] == -1:
            val = df_data['Collision'][-x:].mean()
            arr = np.full(int(b), val)
            ax.plot(x_axis, arr, label="Plan_Rec", color='k')
        else:
            name = namer(d_data[item_ky]['mode'], d_data[item_ky]['option'], d_data[item_ky]['h'])
            ax.plot(x_axis, df_data['Collision'][:b].values, label=name, ls='-')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # ax.set_xscale('log') #{"linear", "log", "symlog", "logit", ...}

    print(df[["ID", "X", "Y", "Z", "h", "o", "m", "MAX_Collision", 'AVG_Collision_{}'.format(x), "episodes",
              "MAX_episodes"]])
    df.to_csv("{}/res.csv".format(father))
    plt.legend()
    plt.savefig("{}/plot.png".format(father))
    plt.show()


def Max_coll(row, data):
    id_ = str(row['ID'])
    data = data[id_]
    return data['df']['Collision'].max()


def uniqueish_color():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())


def avg_coll(row, data, tail):
    id_ = str(row['ID'])
    data = data[id_]
    res = data['df']['Collision'][-tail:].mean()
    return res


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

# "episodes";"Collision";"Wall";"Goal";"stop";"States"
def merge_two_csv(df1,df2):

    diff = abs(len(df1)-len(df2))
    if len(df1)>len(df2):
        small=df2
        big=df1
        small = add_row_n(small,diff)
    elif len(df1)<len(df2):
        big=df2
        small=df1
        small = add_row_n(small, diff)
    elif len(df1)==len(df2):
        big = df2
        small = df1
    else:
        assert False

    assert (len(big) == len(small))
    big.reset_index(drop=True, inplace=True)
    small.reset_index(drop=True, inplace=True)
    res = big.combine(small,np.add)
    assert (len(big) == len(res))
    return res
    # mean_df = pd.concat([big, small]).groupby(level=0).sum()





def get_T(row, data):
    id_ = str(row['ID'])
    data = data[id_]
    res = len(data['df']['Collision'].values)
    return res


def path_to_config(name):
    arr_data = str(name).split("_")
    d = {}
    for item in arr_data:
        if item[0] == "S":
            d["seed"] = item[1:]
        elif item[0] == "I":
            d['id_exp'] = item[1:]
        elif item[0] == "M":
            d['mode'] = int(item[1:])
        elif item[0] == "O":
            d['option'] = int(item[1:])
        elif item[0] == "H":
            d['h'] = int(item[1:])
        elif item[0] == "A":
            d['a'] = int(item[1:])
    d['name'] = '_'.join(str(name).split('_')[:-1])
    return d


def plot_all_csvs(dir_path="/home/eranhe/car_model/debug"):
    res = walk_rec(dir_path, [], "Eval.csv")
    d_l = {}
    max_ep = 0
    for item in res:
        name = str(item).split('/')[-1].split('.')[0]
        d = path_to_config(name)
        df = pd.read_csv(item, sep=';')
        df.dropna(axis=0, inplace=True)
        d['df'] = df
        if max_ep < len(df):
            max_ep = len(df)
        d_l[d["id_exp"]] = d
    return d_l, max_ep


def make_collision_prop(df_l, name):
    # "episodes";"Collision";"Wall";"Goal"
    size = len(df_l)
    if size == 1:
        return
    start_from = []
    for df_i in df_l[:-1]:
        df_i['"Collision"'] = df_i['"Collision"'] / (size - 1)
        start_from.append(df_i['"Collision"'])
    df_subs = pd.concat(start_from)
    y_arr = np.arange(0, len(df_subs), 1)
    plt.plot(y_arr, df_subs.values, label=name, ls=':')
    y_arr = np.arange(len(df_subs), len(df_subs) + len(df_l[-1]['"Collision"'].values), 1)
    # plt.plot(y_arr, df_l[-1]['"Collision"'].values, label=name + "_F", ls='--')
    plt.legend()
    plt.show()

def get_all_exp_path(dir_p="/home/eranhe/car_model/exp/paths"):
    name_dir = os.path.basename(dir_p)
    path_dst = mkdir_system(dir_p,"mean".format(name_dir))
    res = walk_rec(dir_p,[],"_Eval.csv",lv=-1)
    d_exp={}
    for item in res:
        name = os.path.basename(item)
        arr = str(name).split("_")
        idx_exp = int(arr[1][1:])
        if idx_exp not in d_exp:
            d_exp[idx_exp]=[]
        d_exp[idx_exp].append(item)
    for k,v in d_exp.items():
        name = os.path.basename(v[-1])
        l_df=list(map(lambda x: pd.read_csv(x,sep=";"),list(v)))
        max_len = max([len(x) for x in l_df ])
        l_df_mod = list(map(lambda x: add_row_n(x,abs(len(x)-max_len),1000) , l_df))
        v = l_df_mod
        size = len(v)
        ctr=1
        results_df = v[0]

        while ctr<size:
            df_tmp = v[ctr]
            results_df = merge_two_csv(df_tmp,results_df)
            ctr+=1
        results_df = results_df.select_dtypes(exclude=['object', 'datetime'])*1.0/size

        results_df.to_csv(path_dst+"/"+name)
    return path_dst



def tmp(list_csvs,dir_p=None,constant=1000,two=False):
    d = {}
    d_pre_process = {}
    l_baseline = []
    max_ep = 0
    ctr_color = 0
    for item in list_csvs:

        d_info = path_to_config(str(item).split(os.sep)[-1].split('.')[0])

        l_df = read_multi_csvs(item)

        # if l_df[-1]['"Collision"'].iloc[-1]!=1.0:
        #     continue
        pre_process = 0
        for df_i in l_df[:-1]:
            df_i["Collision"] = df_i["Collision"] / (len(l_df) - 1)
            pre_process += len(df_i["Collision"])
        df = pd.concat(l_df[:])

        if d_info['mode'] == -1:
            l_baseline.append(df)
            continue
        if max_ep < len(df):
            max_ep = len(df)

        if d_info['name'] not in d:
            d[d_info['name']] = []
        d[d_info['name']].append(df["Collision"].values)

        d_pre_process[d_info['name']] = pre_process

    for baseline_df in l_baseline:
        val = baseline_df["Collision"].mean()
        arr = np.full(int(max_ep), val)
        arr = arr/constant
        plt.plot(arr, label="Baseline", color='k')

    ky_list = list(d.keys())
    ky_list.sort(key=lambda x: int(str(x).split("_")[1][1:]))
    for j,ky in enumerate(ky_list,1):

        l_coll = d[ky]

        b = np.full([len(l_coll), len(max(l_coll, key=lambda x: len(x)))], 0, dtype=float)
        for i, j_arr in enumerate(l_coll):
            b[i, :len(j_arr)] = j_arr
            if len(j_arr) < b.shape[1]:
                b[i, len(j_arr):] = max(j_arr)

        # b[b<thershold]=0
        b = b/constant
        pre_phase = d_pre_process[ky]
        c = color_array[ctr_color%len(color_array)]
        ctr_color += 1
        name_label = get_name(ky)
        h_name = get_h_name(ky)
        id_exp = get_ID_exp(ky)
        name_label = "{0}({1}) - {2}".format(name_label,h_name,id_exp)

        z = [10e3*x for x in range(len(b.mean(axis=0)))]
        plt.plot(z,b.mean(axis=0)[:], ls='-.', color=c, label=name_label)
        plt.plot(z,b.mean(axis=0)[:], ls='--', color=c)
        if j%3==0 and two:
            plt.title(ky)
            plt.legend()
            if dir_p is not None:
                plt.savefig("{}/{}{}.png".format(dir_p,j, "res"))
            else:
                plt.savefig("{}/{}/{}{}.png".format(os.path.expanduser("~"), "car_model/debug", "res",j))
            plt.show()
            plt.clf()
    # print(ky_uid,"<----")
    plt.legend()
    # save_dir = pt.mkdir_system("{}/car_model/figs".format(expanduser('~')),str(list_csvs[0]).split('/')[-2],False)
    if dir_p is not None:
        plt.savefig("{}/{}.png".format(dir_p,"res"))
    else:
        plt.savefig("{}/{}/{}.png".format(os.path.expanduser("~"), "car_model/debug", "res"))
    plt.show()
    print("end")

def get_h_name(str_name):
    arr = str(str_name).split('_')
    idx_h = int(arr[4][1:])
    if idx_h==8:
        return "$H_{air}$"
    if idx_h==0:
        return "$H_{zero}$"
    return "$H_{}$".format(idx_h)

def get_ID_exp(str_name):
    return str(str_name).split("_")[1][1:]

def get_name(str_name):
    results = "Nan"
    arr = str(str_name).split('_')
    if int(arr[2][1:])==1:
        results="Belief"
    if int(arr[2][1:])==0:
        results = "Position"
    if int(arr[2][1:])==2:
        results = "Position_t"
    return results

def get_state_generated(list_csvs,dir_p=None,two=False):
    d = {}
    d_pre_process = {}
    l_baseline = []
    max_ep = 0
    ctr_color = 0

    # sort by the exp ID
    list_csvs.sort(key=lambda x: int(str(x).split(os.sep)[-1].split("_")[1][1:]))
    # remove the base line
    list_csvs = list(filter(lambda x: True if int(str(x).split(os.sep)[-1].split("_")[2][1:]) != -1 else False,list_csvs))

    for j,item in enumerate(list_csvs):

        d_info = path_to_config(str(item).split(os.sep)[-1].split('.')[0])

        l_df = read_multi_csvs(item)

        # if l_df[-1]['"Collision"'].iloc[-1]!=1.0:
        #     continue
        pre_process = 0
        acc=0
        for df_i in l_df[:]:
            df_i["States"] = df_i["States"] + acc
            acc +=df_i["States"].values[-1]
        df = pd.concat(l_df[:])
        name_label = get_name(d_info["name"])
        h_name = get_h_name(d_info["name"])
        name_label = "{0}({1})".format(name_label,h_name)
        l = df["States"].values
        z = [x*10e3 for x in range(len(l))]
        plt.plot(z,l,label=name_label,color=color_array[ctr_color%len(color_array)],ls="--")
        ctr_color+=1
        if j%2==1 and two:
            plt.legend()
            if dir_p is not None:
                plt.savefig("{}/{}{}.png".format(dir_p,j, "Gen"))
            else:
                plt.savefig("{}/{}/{}{}.png".format(os.path.expanduser("~"), "car_model/debug", "Gen",j))
            plt.show()
            plt.clf()
    plt.legend()
    if dir_p is not None:
        plt.savefig("{}/{}.png".format(dir_p,"Gen"))
    else:
        plt.savefig("{}/{}/{}.png".format(os.path.expanduser("~"), "car_model/debug", "Gen"))
    plt.show()

def just_plot(dir_path="/home/eranhe/car_model/debug"):
    res = walk_rec(dir_path, [], "Eval.csv")
    dico_list = []
    for item in res:
        name = str(item).split('/')[-1].split('.')[0]
        d = path_to_config(name)
        df_list = read_multi_csvs(item)
        for x in df_list:
            x.dropna(axis=0, inplace=True)
        d['df'] = df_list
        dico_list.append(d)
    return dico_list


def tmp33():
    plt.plot(range(0, 10))
    f = 5
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    plt.xlim((xmin * f, xmax * f))
    plt.xlim((ymin * f, ymax * f))
    plt.show()
    exit()


if __name__ == '__main__':
    # tmp33()
    # con_to_dico()
    get_info_path_gen()
    p = "/home/eranhe/car_model/exp/new/h/6"
    p="/home/eranhe/car_model/debug"
    path_dir = '{}/car_model/debug'.format(os.path.expanduser('~'))
    #path_dir = "{}/car_model/exp/intersting/1".format(os.path.expanduser('~'))
    path_dir = get_all_exp_path(p)


    res = walk_rec(path_dir, [], "Eval.csv",lv=-1)
    print(res)
    tmp(res,path_dir,two=False)
    get_state_generated(res,path_dir,two=False)

    get_both_csvs(res,path_dir)
