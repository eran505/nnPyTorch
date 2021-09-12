import pandas as pd

def read_multi_csvs(p):
    big_l = []
    l_csv = []
    sep=','
    with open(p, 'r') as f:
        for line in f:
            line = str(line).replace(" ", "").replace('\n', '').replace('""', '')
            if len(line) < 1:
                continue
            d = {}
            if (str(line).startswith("\"e")) or (str(line).startswith("e")) or (str(line).startswith(",e"))  :
                big_l.append(l_csv)
                l_csv = []
                if str(line).__contains__(";"):
                    sep=';'
                cols = str(line).split(sep)
                continue
            line_arr = str(line).split(sep)
            for i, col in enumerate(cols):
                d[col] = float(line_arr[i])
            l_csv.append(d)
    if (len(l_csv) > 0):
        big_l.append(l_csv)
    df_l = []
    for item in big_l:
        if len(item) > 0:
            df_x = pd.DataFrame(item)
            df_x = df_x.rename(columns={'"Collision"': "Collision",
                                        '"States"': "States"})
            df_l.append(df_x)
    # for item in df_l:
    #     item = rename_col_remove_key(item)
    return df_l

def rename_col_remove_key(df):
    list_col = list(df)
    d = {}
    for ky in list_col:
        d[ky] = str(ky).replace('\"', '')
    return df.rename(columns=d)

if __name__ == '__main__':
    pass