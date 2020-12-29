import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import xgboost as xgb
from os.path import expanduser
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import preprocessor as pr
from sys import exit
from sklearn.preprocessing import LabelEncoder

import joblib


def build_data_set(X,Y,w):
    print("X:",X.shape)


    Y_new = np.argmax(Y,axis=1)

    X_train, X_test, y_train, y_test= train_test_split(X, Y_new, test_size=0.1, random_state=1)


    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    eval_set = [(X_test,y_test)]
    print(X_test.shape)
    print(y_test.shape)

    print("Y:", Y_new.shape)
    n_trees=500

    xgb1 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=n_trees,
        max_depth=5, # [6]
        min_child_weight=1,
        gamma=2, # controls regularization (or prevents overfitting).
        alpha=1, # It controls L1 regularization
        verbosity=1,
        num_class=27,
        use_label_encoder=False,
        subsample=0.8,
        colsample_bytree=0.8, # it control the number of features (variables) supplied to a tree [0.9-05]
        objective='multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    xgb1.fit(X,Y_new)
    #xgb1.fit(X_train, y_train, eval_metric="merror", eval_set=eval_set, verbose=True)
    y_hat_train = xgb1.predict(X_train)
    y_hat_test = xgb1.predict(X_test)
    train_acc = classification_report(y_train,y_hat_train)
    print("train acc: ",train_acc)
    test_acc =  classification_report(y_test,y_hat_test)
    print("test_acc: ",test_acc)
    file_name = "{}/car_model/xgb/xgb_n{}.pt".format(expanduser("~"),n_trees)
    #xgb1.save_model()
    # save model
    joblib.dump(xgb1, file_name)


def get_data():
    np.random.seed(3)
    str_home = expanduser("~")
    if str_home.__contains__('lab2'):
        str_home = "/home/lab2/eranher"

    data_file = "all.csv"

    folder="new/small/30_p"
    folder_dir = "{}/car_model/generalization/{}".format(str_home,folder)
    p_path_data = "{}/{}".format(folder_dir, data_file)
    df = pd.read_csv(p_path_data)

    colz = list(df)


    df = df.loc[df[colz[-1]] > 0]

    df = df.sort_values(by=colz[-1])


    s = len(df)
    df = pr.only_max_value(df,first=True)
    print(len(df),":",s)
    z = df[colz[-2]].value_counts()
    false_count = len(df[colz[-28:-1]]) / df[colz[-28:-1]].sum()

    print(np.sum(z.values))

    print(len(df))

    df.to_csv("{}/car_model/generalization/{}/cut.csv".format(str_home, folder))

    # ax = df[df.columns[-3]].hist()
    # plt.show()

    matrix_df = df.to_numpy()
    build_data_set(matrix_df[:, :-28], matrix_df[:, -28:-1], matrix_df[:, -1])


if __name__ == "__main__":
    get_data()