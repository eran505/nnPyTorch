import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import normalize
from sklearn import preprocessing

import torchvision.datasets as dsets
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from os.path import expanduser
import preprocessor as pr
import helper as hlp
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
from sklearn.datasets import make_regression, make_classification
from my_losses import XTanhLoss,LogCoshLoss,XSigmoidLoss
from fast_data_loader import FastTensorDataLoader
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')




def n_normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def normalize_d(d, target=1.0):
    raw = sum(set(d.values()))
    factor = target / raw
    return {key: value * factor for key, value in d.items()}



class LR(nn.Module):
    def __init__(self, dim, out=27, hidden=400, sec_hidden=300, a=-1.0, b=1.0):
        super(LR, self).__init__()
        # intialize parameters


        self.classifier = nn.Sequential(

            self.make_linear(dim, hidden, a, b),
            nn.BatchNorm1d(hidden),  # applying batch norm
            nn.ReLU(),
            self.make_linear(hidden, hidden, a, b),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            self.make_linear(hidden, sec_hidden, a, b),
            nn.BatchNorm1d(sec_hidden),  # applying batch norm
            nn.ReLU(),
            self.make_linear(sec_hidden, sec_hidden, a, b),
            nn.BatchNorm1d(sec_hidden),  # applying batch norm
            nn.ReLU(),
            self.make_linear(sec_hidden, out, a, b)

        )

    def forward(self, x):
        x=self.classifier(x)
        return x.squeeze()

    def get_weights(self):
        return self.linear1.weight, self.linear2.weight

    def print_weights(self):
        print("linear1_weight=\n{}".format(self.linear1.weight))
        print("linear2_weight=\n{}".format(self.linear2.weight))

    def make_linear(self, in_input, out_output, a_w, b_w):
        layer = torch.nn.Linear(in_input, out_output)
        #torch.nn.init.uniform_(layer.weight, a=a_w, b=b_w)
        #torch.nn.init.uniform_(layer.bias, a=a_w, b=b_w)
        return layer


class NeuralNetwork(object):

    def __init__(self, loss_func, model, optimizer_object):
        self.loss_function = loss_func
        self.nn_model = model
        self.optimizer = optimizer_object
        self.scheduler = optim.lr_scheduler.StepLR(optimizer_object, step_size=4, gamma=0.1)
        self.losses_train = []
        self.losses_test = []
        self.home = None
        self.get_home()
        self.ctr = 0

    def get_home(self):
        str_home = expanduser("~")
        if str_home.__contains__('lab2'):
            str_home = "/home/lab2/eranher"
        self.home = str_home

    def train_step(self, x, y):
        # Sets model to TRAIN mode
        self.ctr += 1
        # Makes predictions
        # print("--Step--")
        # check if req grad = T and where device
        # print(x.requires_grad)
        yhat = self.nn_model(x)

        # for param in self.nn_model.parameters():
        #     print(type(param.data), param.size(),list(param))

        self.optimizer.zero_grad()
        # Computes loss
        loss = self.loss_function(yhat, y.float())  # .detach().float())
        # print("y={0} \nyhat={1}\n".format(y.tolist(),yhat.tolist()))
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()

        # Returns the loss
        return loss.item()

    def fit_model(self, n_epochs, train_dataset, validtion_datatest):
        # For each epoch...

        ctr = 0
        l_time = []
        data_loader_time = []
        loss_tmp = []
        sampels_size_batch = len(train_dataset)
        for epoch in range(n_epochs):
            # Performs one train step and returns the corresponding loss
            training_loader_iter = iter(train_dataset)
            for x_train_tensor, y_train_tensor in training_loader_iter:
                self.nn_model.train()
                t = time.process_time()

                # x_train_tensor, y_train_tensor = next(training_loader_iter)
                data_loader_time.append(time.process_time() - t)
                x_batch = x_train_tensor.to(device)
                y_batch = y_train_tensor.to(device)
                # print(Counter(y_batch.tolist()))
                # print("x_batch={} \t y_batch={}".format(x_batch.requires_grad,y_batch.requires_grad))

                # for auto computing the auto grad
                t = time.process_time()
                loss = self.train_step(x_batch, y_batch)
                l_time.append(time.process_time() - t)

                self.losses_train.append([loss, epoch])
                # if ctr % 1000 == 0:
                #     test_loader_iter = iter(validtion_datatest)
                #     self.eval_nn(test_loader_iter)

                # decay the learning rate
                loss_tmp.append(loss)
                if ctr % 100 == 0:
                    print('Training loss: {2} Iter-{3} Epoch-{0} lr: {1}  Avg-Time:{4} DataLoader(time):{5} '.format(
                        epoch, self.optimizer.param_groups[0]['lr'], np.mean(loss_tmp), ctr / sampels_size_batch,
                        np.mean(l_time), np.mean(data_loader_time)))
                    l_time.clear()
                    data_loader_time.clear()
                    loss_tmp.clear()
                    # print(100 * "-")
                    # print(list(self.nn_model.parameters()))
                ctr = ctr + 1

            self.scheduler.step()
            self.log_to_files()
            torch.save(self.nn_model.state_dict(), "{}/car_model/nn/nn{}.pt".format(self.home,epoch))


    def eval_nn(self, validtion_datatest):
        with torch.no_grad():
            for x_val, y_val in validtion_datatest:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                self.nn_model.eval()

                yhat = self.nn_model(x_val)
                #print("X:{}\t\tY^:{}\t\tY:{}".format(x_val.tolist(),yhat.tolist(),y_val.tolist()))
                val_loss = self.loss_function(y_val, yhat)
                self.losses_test.append(val_loss.item())

                print("test loss= {}".format(val_loss.item()))

    def log_to_files(self):
        hlp.log_file(self.losses_train, "{}/car_model/nn/loss_train.csv".format(self.home), ["loss", "epoch"])
        # hlp.log_file(self.losses_test, "{}/car_model/nn/loss_test.csv".format(self.home), ["loss"])


class DataSet(object):

    def __init__(self, data, targets):
        print("data shape:{}\ntarget shape:{}\n".format(data.shape,targets.shape))
        self.data = data
        self.targets = targets
        self.weights = []
        self.debug_d = None
        self.data = self.norm_without_negative(self.data)
        print(len(self.targets))
        self.targets = self.minmax(self.targets)
        print(len(self.targets))
        print("done")
        self.imbalanced_data_set_weight()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

    def make_dataLoader(self,x,y):
        min_ = [4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0, 8.0, 0.0, -1.0, -1.0, -1.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ptp_ = [7.0, 11.0, 3.0, 8.0, 19.0, 3.0, 11.0, 5.385164807134504, 6.0, 17.0, 3.0, 4.0, 2.0, 2.0, 7.0, 11.0, 3.0, 2.0, 2.0, 2.0, 6.0, 17.0, 3.0, 6.0, 17.0, 3.0, 3.0, 17.0, 3.0, 7.0, 11.0, 3.0, 7.0, 11.0, 3.0, 5.0, 11.0, 3.0]

        x = (x - min_)/ ptp_
        tensor_x = torch.tensor(x, requires_grad=False, dtype=torch.float)
        tensor_y = torch.tensor(y)
        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        my_dataloader = DataLoader(my_dataset, shuffle=True, batch_size=len(y), num_workers=0)
        return my_dataloader

    def minmax(self,foo):
        return preprocessing.minmax_scale(foo, feature_range=(-1, 1))
    def norm(self):
        self.data = normalize(self.data, axis=0, norm='l1')

    def to_bin(self, occurrence_matrix, bins=None):
        d_bin = {}
        d_bin_occ = {}
        ctr = 0
        elements_array = occurrence_matrix[:, 0]
        sum_all = elements_array.sum()
        if bins is None:
            bins = np.arange(np.min(elements_array), np.max(elements_array), 0.1)
        for i in range(1, 100):
            z = elements_array[np.digitize(elements_array, bins) == i]
            if ctr == len(elements_array):
                break
            if len(z) == 0:
                continue
            for e in z:
                d_bin[e] = {"bin": i, "occ": occurrence_matrix[ctr, 1], 'w': 0}
                if i not in d_bin_occ:
                    d_bin_occ[i] = 0
                d_bin_occ[i] += occurrence_matrix[ctr, 1]
                ctr = ctr + 1
        dixt_W = {}
        for ky in d_bin:
            d_bin[ky]['w'] = sum_all / d_bin_occ[d_bin[ky]['bin']]
            dixt_W[ky] = d_bin[ky]['w']
        print(d_bin_occ)
        print("d_bin_occ")
        dixt_W[1.0] = dixt_W[1.0]
        return normalize_d(dixt_W)

    def imbalanced_data_set_weight(self, bins=None):
        # if bins is None:
        #     bins = [0, 0.001, 1.1]
        labels = self.targets
        labels_avg = labels.mean(1)
        print(labels_avg.shape)
        print("labels {}".format(len(labels)))
        unique, counts = np.unique(labels.mean(1), return_counts=True)
        unique_arr = np.asarray((unique, counts)).T
        #W_bin = self.to_bin(unique_arr, bins)
        sum = unique_arr[:, 1].sum()
        W = sum / unique_arr[:, 1]
        W = n_normalize(W)
        pairs = zip(unique_arr[:, 0], W)
        d_all_w = {i[0]: i[1] for i in pairs}
        a = np.vectorize(d_all_w.get)(labels_avg) # or W_bin --> d_all_w
        self.weights = a
        #self.debug_d = result

    def norm_without_negative(self,table_data):
        print("####"*50)
        print(list(table_data.min(0)))
        print(list(table_data.ptp(0)))
        print("####"*50)
        return (table_data - table_data.min(0)) / table_data.ptp(0)

    def split_test_train(self,raito=0.001):
        self.weights=np.ones(self.data.shape[0])
        X_train, X_test, y_train, y_test, w_train, w_test = \
            train_test_split(self.data, self.targets, self.weights, test_size=raito, random_state=0)
        loader_train = self.make_DataSet(X_train, y_train, size_batch=batch_size, is_shuffle=False,
                                         samples_weights=w_train)
        # l = torch.multinomial(torch.tensor(w_test),len(w_test),False).tolist()

        loader_test = self.make_DataSet(X_test, y_test, size_batch=len(y_test), samples_weights=w_test)

        return loader_train, loader_test

    def make_DataSet(self, X_data, y_data, size_batch=1, is_shuffle=False, samples_weights=None,pin_memo = False):
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True)
        print("-----------batch size = {}".format(size_batch))
        if device.type != 'cpu':
            pin_memo = True

        tensor_x = torch.tensor(X_data, requires_grad=False, dtype=torch.float)
        tensor_y = torch.tensor(y_data)
        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        my_dataloader = DataLoader(my_dataset, shuffle=is_shuffle, batch_size=size_batch, num_workers=0
                                    , sampler=sampler,pin_memory=pin_memo)  # ),)  # create your dataloader
        return my_dataloader


def main(in_dim, train_dataset, test_dataset):
    print(device)
    SEED = 2809
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    ## hyperparams
    num_iterations = 20
    lrmodel = LR(in_dim)
    lrmodel = lrmodel.to(device)

    loss = XSigmoidLoss()
    # SGD/Adam
    optimizer = torch.optim.Adam(lrmodel.parameters(), lr=0.01)

    my_nn = NeuralNetwork(loss_func=loss,
                          optimizer_object=optimizer,
                          model=lrmodel)

    my_nn.fit_model(num_iterations, train_dataset, test_dataset)


batch_size = 64


import pandas as pd
if __name__ == "__main__":

    start = time.time()
    # x, y = pr.MainLoader()
    end = time.time()
    print("MainLoader Time: {}".format(end - start))
    df = pd.read_csv("/home/eranhe/car_model/generalization/split_data/all.csv")
    matrix_df = df.as_matrix()

    # x,y = make_classification(n_samples=1000000,n_features=16,n_informative=8,n_classes=2)
    # x = x[number:number * 2]
    # y = y[number:number * 2]
    DataLoder = DataSet(matrix_df[:, :-27], matrix_df[:, -27:])
    train_loader,_ = DataLoder.split_test_train(0.000001)
    _, test_loader = DataLoder.split_test_train(0.1)

    # train_loader = FastTensorDataLoader(torch.tensor(x[:-100]).float(),torch.tensor(y[:-100]).float(),batch_size=4)
    # test_loader  = FastTensorDataLoader(torch.tensor(x[-100:]).float(), torch.tensor(y[-100:]).float(), batch_size=4)
    # df = pd.read_csv("/home/eranhe/car_model/generalization/first_stateV2.csv",sep='\t',index_col=0)
    # df= df.as_matrix()
    # first_state_loader = DataLoder.make_dataLoader(df[:,:-27],df[:,-27:])
    print(len(train_loader))
    print(len( test_loader))
    main(matrix_df.shape[-1]-27, train_loader, test_loader)
