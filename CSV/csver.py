from os_util import walk_rec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from CSV.utils_csver import read_multi_csvs

# "episodes";"Collision";"Wall";"Goal"

color_array = ['red', 'green', 'blue', 'orange', 'gray', "yellow", "brown", "purple", "m"]


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


def tmp(list_csvs):
    d = {}
    d_pre_process = {}
    l_baseline = []
    max_ep = 0
    ctr_color = 0;
    for item in list_csvs:

        d_info = path_to_config(str(item).split(os.sep)[-1].split('.')[0])

        l_df = read_multi_csvs(item)

        # if l_df[-1]['"Collision"'].iloc[-1]!=1.0:
        #     continue
        pre_process = 0
        for df_i in l_df[:-1]:
            df_i['"Collision"'] = df_i['"Collision"'] / (len(l_df) - 1)
            pre_process += len(df_i['"Collision"'])
        df = pd.concat(l_df[:])

        if d_info['mode'] == -1:
            l_baseline.append(df)
            continue
        if max_ep < len(df):
            max_ep = len(df)

        if d_info['name'] not in d:
            d[d_info['name']] = []
        d[d_info['name']].append(df['"Collision"'].values)

        d_pre_process[d_info['name']] = pre_process

    for baseline_df in l_baseline:
        val = baseline_df['"Collision"'].mean()
        arr = np.full(int(max_ep), val)
        plt.plot(arr, label="Plan_Rec", color='k')
    for ky in d:

        l_coll = d[ky]

        b = np.full([len(l_coll), len(max(l_coll, key=lambda x: len(x)))], 0, dtype=float)
        for i, j_arr in enumerate(l_coll):
            b[i, :len(j_arr)] = j_arr
            if len(j_arr) < b.shape[1]:
                b[i, len(j_arr):] = max(j_arr)

        # b[b<thershold]=0

        pre_phase = d_pre_process[ky]
        c = color_array[ctr_color]
        ctr_color += 1
        plt.plot(b.mean(axis=0)[:], ls=':', color=c, label=ky)
        plt.plot(b.mean(axis=0)[:pre_phase], ls='-', color=c)
    # print(ky_uid,"<----")
    plt.legend()
    # save_dir = pt.mkdir_system("{}/car_model/figs".format(expanduser('~')),str(list_csvs[0]).split('/')[-2],False)
    plt.savefig("{}/{}/{}.png".format(os.path.expanduser("~"),"car_model/debug","res"))
    plt.show()
    print("end")


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
    # exit()
    res = walk_rec('{}/car_model/debug'.format(os.path.expanduser('~')), [], "Eval.csv")
    print(res)
    tmp(res)

    # dico_d = just_plot()
    # for item in dico_d:
    #     make_collision_prop(item['df'],item['id_exp'])
