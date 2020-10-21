import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import pandas as pd
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
from my_losses import XTanhLoss, LogCoshLoss, XSigmoidLoss
from fast_data_loader import FastTensorDataLoader


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')


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
    def __init__(self, dim, out=27, hidden=300, sec_hidden=300, a=-1.0, b=1.0):
        super(LR, self).__init__()
        # intialize parameters

        self.classifier = nn.Sequential(

            self.make_linear(dim, hidden, a, b),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden),  # applying batch norm
            self.make_linear(hidden, hidden, a, b),
            # #nn.BatchNorm1d(hidden),
            nn.ReLU(),
            self.make_linear(hidden, sec_hidden, a, b),
            # #nn.BatchNorm1d(sec_hidden),  # applying batch norm
            nn.ReLU(),
            # self.make_linear(sec_hidden, sec_hidden, a, b),
            # nn.Tanh(),
            #nn.BatchNorm1d(sec_hidden),  # applying batch norm
            self.make_linear(sec_hidden, out, a, b)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x.squeeze()

    def get_weights(self):
        return self.linear1.weight, self.linear2.weight

    def print_weights(self):
        print("linear1_weight=\n{}".format(self.linear1.weight))
        print("linear2_weight=\n{}".format(self.linear2.weight))

    def make_linear(self, in_input, out_output, a_w, b_w):
        layer = torch.nn.Linear(in_input, out_output)
        # torch.nn.init.uniform_(layer.weight, a=a_w, b=b_w)
        # torch.nn.init.uniform_(layer.bias, a=a_w, b=b_w)
        return layer


class NeuralNetwork(object):

    def __init__(self, loss_func, model, optimizer_object):
        self.loss_function = loss_func
        self.nn_model = model
        self.optimizer = optimizer_object
        self.scheduler = optim.lr_scheduler.StepLR(optimizer_object, step_size=3, gamma=0.1)
        self.losses_train = []
        self.losses_test = []
        self.home = None
        self.get_home()
        self.ctr = 0
        self.debug_D = {}

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
        #y = nn.Softmax(y)
        loss = self.loss_function(yhat,y )#)reduction="sum")  # .detach().float())

        #print("y={0} \nyhat={1}\n".format(y.tolist(),yhat.tolist()))
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()

        # Returns the loss
        return loss.item()

    def fit_model(self, n_epochs, train_dataset, validtion_datatest=None):
        # For each epoch...
        ctr = 0
        l_time = []
        data_loader_time = []
        loss_tmp = []
        sampels_size_batch = len(train_dataset)
        for epoch in range(n_epochs):
            # Performs one train step and returns the corresponding loss
            training_loader_iter = iter(train_dataset)
            ctr=0
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
                losser=[]

                loss = self.train_step(x_batch,y_batch)

                l_time.append(time.process_time() - t)

                self.losses_train.append([loss, epoch])
                if ctr % 10000 == 0:
                    test_loader_iter = iter(validtion_datatest)
                    self.eval_nn(test_loader_iter)

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
            torch.save(self.nn_model.state_dict(), "{}/car_model/nn/nn{}.pt".format(self.home, epoch))

    def eval_nn(self, validtion_datatest):
        self.losses_test=[]
        with torch.no_grad():
            for x_val, y_val in validtion_datatest:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                self.nn_model.eval()

                yhat = self.nn_model(x_val)
                # print("X:{}\t\tY^:{}\t\tY:{}".format(x_val.tolist(),yhat.tolist(),y_val.tolist()))
                val_loss = self.loss_function(y_val, yhat)
                self.losses_test.append(val_loss.item())
                break
            print("test loss= avg:{}  max:{}".format(sum(self.losses_test) / len(self.losses_test),max(self.losses_test)))

    def log_to_files(self):
        hlp.log_file(self.losses_train, "{}/car_model/nn/loss_train.csv".format(self.home), ["loss", "epoch"])
        # hlp.log_file(self.losses_test, "{}/car_model/nn/loss_test.csv".format(self.home), ["loss"])
    def log_dict_debug(self,ep):
        print("debug_d = ", len(self.debug_D))
        with open('/home/ERANHER/car_model/generalization/test.csv', 'w') as f:
            for key in self.debug_D.keys():
                f.write("%s,%s\n" % (key, self.debug_D[key]/ep))


class DataSet(object):

    def __init__(self, data, targets,W=[]):
        print("data shape:{}\ntarget shape:{}\n".format(data.shape, targets.shape))
        self.data = data
        self.targets = targets
        self.weights = W
        self.debug_d = None
        self.data = self.norm_without_negative(self.data)
        print(len(self.targets))
        self.targets = self.min_max_zero_to_one(self.targets)
        print(len(self.targets))
        print("done")
        if len(W)==0:
            self.imbalanced_data_set_weight()
        else:
            self.weights = normalize(self.weights[:,np.newaxis], axis=0).ravel()


    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

    def scale_negtive_one_to_one(self, foo):
        return preprocessing.maxabs_scale(foo)

    def min_max_zero_to_one(self,foo):
        return preprocessing.minmax_scale(foo)

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
        # W_bin = self.to_bin(unique_arr, bins)
        sum = unique_arr[:, 1].sum()
        W = sum / unique_arr[:, 1]
        W = n_normalize(W)
        pairs = zip(unique_arr[:, 0], W)
        d_all_w = {i[0]: i[1] for i in pairs}
        a = np.vectorize(d_all_w.get)(labels_avg)  # or W_bin --> d_all_w
        self.weights = a
        # self.debug_d = result

    def norm_without_negative(self, table_data):
        print("####" * 50)
        min_arr = table_data.min(0)
        ptp_arr = table_data.ptp(0)
        print(list(min_arr))
        print(list(ptp_arr))

        print("####" * 50)
        return (table_data - min_arr) / ptp_arr

    def split_test_train(self, raito=0.001):
        self.weights = np.ones(self.data.shape[0])
        X_train, X_test, y_train, y_test, w_train, w_test = \
            train_test_split(self.data, self.targets, self.weights, test_size=raito, random_state=0)
        loader_train = DataSet.make_DataSet(X_train, y_train, size_batch=batch_size, is_shuffle=False,
                                            samples_weights=w_train)
        # l = torch.multinomial(torch.tensor(w_test),len(w_test),False).tolist()

        loader_test = DataSet.make_DataSet(X_test, y_test, size_batch=16, samples_weights=w_test)

        return loader_train, loader_test

    @staticmethod
    def make_DataSet(X_data, y_data, size_batch=1, is_shuffle=False, samples_weights=None
                     , pin_memo=False,over_sample=True):
        sampler=None
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True)
        print("-----------batch size = {}".format(size_batch))
        if device.type != 'cpu':
            pin_memo = True

        tensor_x = torch.tensor(X_data, requires_grad=False, dtype=torch.float64)
        tensor_y = torch.tensor(y_data, dtype=torch.float64).contiguous()
        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        if over_sample:
            return DataLoader(my_dataset, shuffle=is_shuffle, batch_size=size_batch, num_workers=0
                                   , sampler=sampler, pin_memory=pin_memo)  # ),)  # create your dataloader
        else:
            return DataLoader(my_dataset, shuffle=is_shuffle, batch_size=size_batch, num_workers=0)





def main(in_dim, train_dataset, test_dataset=None):
    print(device)
    SEED = 2809
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    ## hyperparams
    num_iterations = 30
    lrmodel = LR(in_dim).double()
    lrmodel = lrmodel.to(device)

    #loss = nn.NLLLoss
    #loss = nn.functional.kl_div
    #loss= nn.KLDivLoss()
    #loss = XSigmoidLoss()
    loss=F.mse_loss
    # SGD/Adam
    optimizer = torch.optim.SGD(lrmodel.parameters(), lr=0.0955,momentum=0.5)

    my_nn = NeuralNetwork(loss_func=loss,
                          optimizer_object=optimizer,
                          model=lrmodel)

    # hlp.get_the_best_lr(lrmodel,loss,train_dataset)
    # exit()
    my_nn.fit_model(num_iterations, train_dataset, test_dataset)


def test_main(path_to_model):
    df = pd.read_csv("/home/eranhe/car_model/generalization/4data/nn_DATA/all.csv")
    matrix_df = df.to_numpy()  # [756251:756254]
    print(len(matrix_df[:, :-27]))
    print(matrix_df)
    obj = DataSet(matrix_df[:, :-27], matrix_df[:, -27:])
    test_loader = DataSet.make_DataSet(obj.data, obj.targets, size_batch=1)
    in_p = matrix_df.shape[-1] - 27
    my_model = LR(in_p).double()

    my_model.load_state_dict(torch.load(path_to_model, map_location=device))
    my_model.cpu()
    # self.nn = self.nn.double()
    my_model.eval()
    sum = 0
    ctr = 0
    with torch.no_grad():
        for x_val, y_val in iter(test_loader):
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            my_model.eval()
            yhat = my_model(x_val)
            k=0
            for i,j in list(zip(yhat, y_val.squeeze())):
                print("{}| Y^:{} | Y:{}".format(k,i,j))
                k+=1

            sum += F.smooth_l1_loss(y_val.squeeze(), yhat).item()
            if ctr % 1000 == 0:
                print("{}".format(ctr))
            ctr += 1
            print("---- losses --------")
            print("MAE: ",F.l1_loss(y_val.squeeze(), yhat).item())
            print("MSE: ", F.mse_loss(y_val.squeeze(), yhat).item())
            exit()
    print("SUM -> ", sum)
    exit()


batch_size = 64

# 756253:756251 index



if __name__ == "__main__":
    np.random.seed(3)
    str_home = expanduser("~")
    if str_home.__contains__('lab2'):
        str_home = "/home/lab2/eranher"

    #test_main("{}/car_model/nn/nn15.pt".format(str_home))

    start = time.time()
    # x, y = pr.MainLoader()
    end = time.time()
    df = pd.read_csv("{}/car_model/generalization/5data/all.csv".format(str_home))
    print(len(df))
    #df = df.sample(frac=1).reset_index(drop=True)
    print(len(df))
    #df = pr.only_max_value(df)
    matrix_df = df.to_numpy()


    print(len(list(df)))

    test_loader=None
    DataLoder = DataSet(matrix_df[:, :-28], matrix_df[:, -28:-1],matrix_df[:,-1])
    train_loader, _ = DataLoder.split_test_train(0.0000001)

    df = pd.read_csv("{}/car_model/generalization/7data/all.csv".format(str_home))
    df = pr.only_max_value(df)
    matrix_df = df.to_numpy()
    DataLoder = DataSet(matrix_df[:, :-28], matrix_df[:, -28:-1],matrix_df[:,-1])
    _, test_loader = DataLoder.split_test_train(0.1)



    print("len - train_loader:",len(train_loader))

    main(matrix_df.shape[-1] - 28, train_loader, test_loader)
