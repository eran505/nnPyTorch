import os_util as pt
import pandas as pd
import os
import matplotlib.pyplot as plt
from os.path import expanduser
from itertools import groupby


def get_all_df_sort_by_size(p_path="/home/eranhe/car_model/debug"):
    res = pt.walk_rec(p_path, [], ".csv")
    d_path = {}
    d_size = {}
    for item_csv in res:
        ky = str(item_csv).split('/')[-1].split('.')[0]
        if ky == "d_states":
            continue
        size_i = os.path.getsize(item_csv)
        d_path[ky] = item_csv
        d_size[ky] = size_i
    print(d_path)
    print(d_size)
    sort_tuples = sorted(d_size.items(), key=lambda x: x[1], reverse=True)
    sort_tuples_all = []
    for k, v in sort_tuples:
        sort_tuples_all.append((k, v, d_path[k]))
    print(sort_tuples_all)
    return sort_tuples_all


def filter_top_k(k, df):
    d = {}
    for i in list(range(0, 27)):
        d[str(i)] = df[str(i)].iloc[-1]
    sort_tuples = sorted(d.items(), key=lambda x: x[1], reverse=True)
    sort_tuples_cut = sort_tuples[:k]
    res = [k for k, v in sort_tuples_cut]
    return res


def plot_graph(list_sorted_tuple):
    list_col_str = [str(x) for x in list(range(0, 27))]
    for key, size, path in list_sorted_tuple:
        print("{} {} {}".format("-" * 10, key, "-" * 10))
        df = pd.read_csv(path, names=list_col_str)
        list_col = filter_top_k(10, df)
        df[list_col].plot()
        plt.show()


def get_the_last_row(path_p, col_name="Collision"):
    col = "Collision"
    res = pt.walk_rec(path_p, [], "Eval.csv")
    d_list = []
    for item in res:
        print(item)
        name = "_".join(str(item).split('/')[-1].split('.')[0].split('_')[:-1])
        seedID = int(str(name).split("_")[0])
        configID = int(str(name).split("_")[1][1:])
        abstraction = int(str(name).split("_")[2][1:])
        df_i = pd.read_csv(item, sep="\s+|;|,")
        if len(df_i) < 2:
            continue
        df_i.dropna(inplace=True)
        df_i = rename_col_remove_key(df_i)
        df_i = df_i.tail(1)
        last_row = df_i.mean(axis=0).to_dict()
        last_row["seed"] = seedID
        last_row["config"] = configID
        last_row["Max Abstraction"] = abstraction
        d_list.append(last_row)
        print(last_row)
    newDF = pd.DataFrame(d_list)
    # df_new = newDF.groupby(['config', 'Max Abstraction'])[col].agg('mean').reset_index()
    #
    # l = list(df_new['config'].unique())
    # d={}
    # for x in l:
    #     d[x]=pow(2,x-1)
    # df_new['config'] = df_new['config'].map(d)
    # df_new = df_new[df_new['config'] < 40]
    df = newDF.pivot(index='config', columns='Max Abstraction', values=col)
    ax = df.plot()
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel('{} Rate'.format(col))
    # ax.legend(loc='center right')
    #
    # newDF.plot(kind='line',x="config",y="Collision")
    plt.savefig('{}/plot.png'.format(path_p))  # save the figure to file
    plt.show()


def rename_col_remove_key(df):
    list_col = list(df)
    d = {}
    for ky in list_col:
        d[ky] = str(ky).replace('\"', '')
    return df.rename(columns=d)


def all_policy_eval(path_p, col_name="Collision"):  ##"ctr_round","ctr_wall","ctr_coll","ctr_at_goal"
    res = pt.walk_rec(path_p, [], "Eval.csv")
    d = {}
    df_list = []
    newDF = pd.DataFrame()
    for item in res:
        name = "_".join(str(item).split('/')[-1].split('.')[0].split('_')[:-1])
        df_i = pd.read_csv(item, sep=';')
        print(list(df_i))
        df_list.append(df_i)
        newDF["{}-{}".format(name, col_name)] = df_i[col_name] / 500

    newDF.plot(kind='line')
    plt.savefig('{}/plot.png'.format(path_p))  # save the figure to file
    plt.show()


def merge_all_policy_eval(path_p):
    res = pt.walk_rec(path_p, [], "Eval.csv")
    d = {}
    df_list = []
    for item in res:
        df_list.append(pd.read_csv(item))
    df_all = pd.concat(df_list).groupby(level=0).mean()
    # dfObj = df_all.transpose()
    print(list(df_all))
    l_col = ['ctr_wall', 'ctr_coll', 'ctr_at_goal', 'ctr_open']
    l_col_ratio = [("_".join(str(x).split('_')[1:]) + "_ratio").title() for x in l_col]
    for col, col_new in zip(l_col, l_col_ratio):
        df_all[col_new] = df_all[col] / 500
    df_all = df_all.set_index('ctr_round')
    df_all[l_col_ratio].plot(kind='line')
    plt.savefig('{}/plot.png'.format(path_p))  # save the figure to file
    plt.show()


def coundEndPoint(dir_path, filter_key='_8'):
    res = pt.walk_rec(p, [], ".csv")
    for item in res:
        if str(item).split('/')[-1].__contains__(filter_key) is False:
            continue


def trajectory_read(p_path):
    df = pd.read_csv(p_path, names=['tran'])
    print(list(p_path))
    idx = (df['tran'] == "END").idxmax()
    print(idx)
    df = df.iloc[idx + 1:]
    array = df['tran'].to_numpy()
    res = [list(g) for k, g in groupby(array, lambda x: x != "END") if k]
    return res


def read_multi_csvs(p):
    big_l = []
    l_csv = []
    with open(p, 'r') as f:
        for line in f:
            line = str(line).replace(" ", "").replace('\n', '').replace('""', '')
            if len(line) < 1:
                continue
            d = {}
            if (str(line).startswith("\"e")):
                big_l.append(l_csv)
                l_csv = []
                cols = str(line).split(';')
                continue
            line_arr = str(line).split(";")
            for i, col in enumerate(cols):
                d[col] = float(line_arr[i])
            l_csv.append(d)
    if (len(l_csv) > 0):
        big_l.append(l_csv)
    df_l = []
    for item in big_l:
        if len(item) > 0:
            df_l.append(pd.DataFrame(item))

    return df_l


def one_path_ana(path_p):
    print(path_p)
    father = "/".join(str(path_p).split("/")[:-1])
    name = str(path_p).split("/")[-1].split(".")[0]
    l = []
    res = pt.walk_rec(path_p, [], "Eval.csv")
    print("res:", len(res))
    for item_csv_p in res:
        d = {}
        print(item_csv_p)
        d["seed_number"] = str(item_csv_p).split("/")[-1].split("_")[0]
        d["u_id"] = str(item_csv_p).split("/")[-1].split("_")[1]
        d["learn_mode"] = str(item_csv_p).split("/")[-1].split("_")[2]
        d["df_l"] = read_multi_csvs(item_csv_p)
        d["reward"] = str(name).split("r_")[-1][0]
        d["shaffle"] = str(name).split("s_")[-1][0]
        df_pre = mean_multi_dfs(d["df_l"][:-1])
        last_row = d["df_l"][-1].tail(1).mean(axis=0).to_dict()
        for ky in last_row:
            d["{}".format(ky).replace('\"', '')] = last_row[ky]
        d['path_size'] = len(d["df_l"]) - 1
        for ky in df_pre:
            d["{}_{}".format("P", ky).replace('\"', '')] = df_pre[ky]
        del d["df_l"]
        l.append(d)
    df = pd.DataFrame(l)
    df["P_episodes"]=df["P_episodes"].fillna(0)
    df["SUM_episodes"] = df['episodes'] + df['P_episodes']
    df.to_csv("{}/{}.csv".format(father, name))
    return df


def mean_multi_dfs(l_df, tail_row=1):
    means_arr = []
    for df in l_df:
        tail_df = df.tail(tail_row)
        mean = tail_df.mean(axis=0)
        means_arr.append(mean)
    df_all = pd.DataFrame(means_arr)
    df_mean = df_all.sum(axis=0)
    return df_mean.to_dict()


def one_path(path_p="/home/eranhe/car_model/out"):
    print(path_p)
    l = []
    res = pt.walk_rec(path_p, [], "Eval.csv")

    for item_csv_p in res:
        d = {}
        print(item_csv_p)
        d["seed_number"] = str(item_csv_p).split("/")[-1].split("_")[0]
        d["u_id"] = str(item_csv_p).split("/")[-1].split("_")[1]
        d["max_L"] = str(item_csv_p).split("/")[-1].split("_")[2]
        df = pd.read_csv(item_csv_p, sep=';')
        df = df.dropna()
        df_tail = df.tail(1)
        last_row = df_tail.mean(axis=0).to_dict()
        for ky in last_row:
            d["{}".format(ky).replace('\"', '')] = last_row[ky]
        l.append(d)
    df = pd.DataFrame(l)
    df.to_csv("{}/all_one_path.csv".format(path_p))
    return df


def one_vs_all(int_dix):
    sub_dir = "car_model/singal"
    home = expanduser("~")
    dir_arr = ["one_path", "all_path"]
    dir_name = dir_arr[int_dix]
    res = pt.walk_rec("{}/{}/{}".format(home, sub_dir, dir_name), [], "", False, lv=-1)
    for item in res:
        print(res)
        df1 = one_path_ana(item)

    res = pt.walk_rec("{}/{}/{}".format(home, sub_dir, dir_name), [], ".csv", lv=-1)
    df_all = None
    for df_path in res:
        df_i = pd.read_csv(df_path, index_col=0)
        if df_all is None:
            df_all = df_i
        else:
            df_all = df_all.append(df_i)
    df_all.to_csv("{}/{}/{}/{}".format(home, sub_dir, dir_name, 'all.csv'))


def play_data(df_all_path, df_one_path):
    df_all_at_once = pd.read_csv(df_all_path, index_col=0)
    df_one_by_one = pd.read_csv(df_one_path, index_col=0)
    print(len(df_all_at_once))
    print(len(df_one_by_one))
    del df_all_at_once['path_size']
    df_paths = df_one_by_one[['seed_number', 'u_id', 'path_size']]
    new_df_all = pd.merge(df_all_at_once, df_paths, on=['seed_number', 'u_id'], how="inner")
    print(list(new_df_all))
    df_one_by_one['merge_episodes'] = df_one_by_one['episodes'] + df_one_by_one['P_episodes']
    print(list(df_one_by_one))
    df_one = df_one_by_one[
        ['seed_number', 'u_id', 'merge_episodes', 'episodes', 'Collision', 'path_size', 'P_Collision']]
    df_all = new_df_all[['seed_number', 'u_id', 'episodes', 'Collision', 'path_size']]
    df = pd.merge(df_one, df_all, on=['seed_number', 'u_id', 'path_size'], how="inner", suffixes=["_one", '_all'])
    print(" df size: ", len(df))
    print("one size: ", len(df_one))
    assert (len(df) == len(df_one))
    assert (len(df) == len(df_all))
    print(list(df))
    father = '/'.join(str(df_all_path).split('/')[:-1])
    df.to_csv("{}/df.csv".format(father))


def get_data_from_exp(path_to_df):
    all_df = pd.read_csv(path_to_df, index_col=0)

    print(list(all_df))


if __name__ == "__main__":
    print("hello world")
    one_path_ana("/home/eranhe/car_model/out")
    exit()
    one_path_ana("/home/ise/car_model/out/eranh")
    one_vs_all(0)
    one_vs_all(1)
    play_data("/home/ERANHER/car_model/singal/all_path/all.csv",
              "/home/ERANHER/car_model/singal/one_path/all.csv")

    p = "/home/ERANHER/car_model/results/26_04/con1"
    #p_path = "/home/ERANHER/car_model/results/dataEXP/old/sizeExp/roni/out"
    p_path = "/home/eranhe/car_model/ABS/size__p_5"
    # all_policy_eval(p_path,"ctr_at_goal") #ctr_at_goal  ctr_open
    get_the_last_row(p_path)
    # merge_all_policy_eval()
    # res = merge_all_policy_eval(p)
