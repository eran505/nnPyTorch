import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import normalize
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


def n_normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def normalize_d(d, target=1.0):
    raw = sum(set(d.values()))
    factor = target / raw
    return {key: value * factor for key, value in d.items()}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class LR(nn.Module):
    def __init__(self, dim):
        super(LR, self).__init__()
        # intialize parameters
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = F.relu

    def forward(self, x):
        x = self.linear1(x)
        #x = self.relu(x)
        self.relu(x, inplace=True)
        x = self.linear2(x)
        #x = self.relu(x)
        self.relu(x,inplace=True)
        #x = self.sigmoid(x)
        # x = self.sigmoid(x)
        return x.squeeze()


class NeuralNetwork(object):

    def __init__(self, loss_func, model, optimizer_object):
        self.loss_function = loss_func
        self.nn_model = model
        self.optimizer = optimizer_object
        self.scheduler = optim.lr_scheduler.StepLR(optimizer_object, step_size=1000000, gamma=0.1)
        self.losses_train = []
        self.losses_test = []
        self.home = expanduser("~")

    def train_step(self, x, y):
        # Sets model to TRAIN mode
        self.nn_model.train()
        # Makes predictions

        # check if req grad = T and where device

        yhat = self.nn_model(x)
        # Computes loss
        loss = self.loss_function(y, yhat)
        # print("y={0} \nyhat={1}\n".format(y.tolist(),yhat.tolist()))
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    def fit_model(self, n_epochs, train_dataset, validtion_datatest):
        # For each epoch...
        ctr = 0
        for epoch in range(n_epochs):
            # Performs one train step and returns the corresponding loss

            training_loader_iter = iter(train_dataset)
            test_loader_iter = iter(validtion_datatest)
            for x_train_tensor, y_train_tensor in training_loader_iter:
                # x_train_tensor, y_train_tensor = next(training_loader_iter)
                x_batch = x_train_tensor.to(device)
                y_batch = y_train_tensor.to(device)
                # print(Counter(y_batch.tolist()))
                # print("x_batch={} \t y_batch={}".format(x_batch.requires_grad,y_batch.requires_grad))

                # for auto computing the auto grad
                # x_batch.requires_grad_()
                t = time.process_time()
                loss = self.train_step(x_batch, y_batch)
                print(time.process_time() - t)

                self.losses_train.append([loss, epoch])

                # if epoch % 10000 == 0:
                #     self.eval_nn(test_loader_iter)

                # decay the learning rate
                self.scheduler.step()

                print('Training loss: {2} Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr'], loss))


            self.log_to_files()
        torch.save(self.nn_model.state_dict(),"{}/car_model/nn/nn.pt".format(self.home))

    def eval_nn(self, validtion_datatest):
        with torch.no_grad():
            for x_val, y_val in validtion_datatest:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                self.nn_model.eval()

                yhat = self.nn_model(x_val)

                print(yhat.shape)
                val_loss = self.loss_function(y_val, yhat)
                self.losses_test.append(val_loss.item())

                print("test loss= {}".format(val_loss.item()))

    def log_to_files(self):
        hlp.log_file(self.losses_train, "{}/car_model/nn/loss_train.csv".format(self.home), ["loss", "epoch"])
        #hlp.log_file(self.losses_test, "{}/car_model/nn/loss_test.csv".format(self.home), ["loss"])


class DataSet(object):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.weights = None
        self.debug_d = None
        self.norm_without_negative()
        self.imbalanced_data_set_weight()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

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
        return normalize_d(dixt_W)

    def imbalanced_data_set_weight(self, bins=None):
        if bins is None:
            bins = [0, 0.001, 1.1]
        labels = self.targets
        print("labels {}".format(len(labels)))
        unique, counts = np.unique(labels, return_counts=True)
        unique_arr = np.asarray((unique, counts)).T
        W_bin = self.to_bin(unique_arr, bins)
        sum = unique_arr[:, 1].sum()
        W = sum / unique_arr[:, 1]
        W = n_normalize(W)
        pairs = zip(unique_arr[:, 0], W)
        result = {i[0]: i[1] for i in pairs}
        print(result)
        a = np.vectorize(W_bin.get)(labels)
        self.weights = a
        self.debug_d = result

    def norm_without_negative(self):
        print(self.data.min(0))
        print(self.data.ptp(0))
        self.data = (self.data - self.data.min(0)) / self.data.ptp(0)

    def split_test_train(self):
        X_train, X_test, y_train, y_test, w_train, w_test = \
            train_test_split(self.data, self.targets, self.weights, test_size=0.00001, random_state=0)
        loader_train = self.make_DataSet(X_train, y_train, size_batch=16, is_shuffle=False,
                                         samples_weights=w_train)

        loader_test = self.make_DataSet(X_test, y_test, size_batch=len(X_test), samples_weights=w_test)

        return loader_train, loader_test

    def make_DataSet(self, X_data, y_data, size_batch=1, is_shuffle=False, samples_weights=None):
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True)
        tensor_x = torch.tensor(X_data, requires_grad=False, dtype=torch.float, device=device)
        tensor_y = torch.tensor(y_data)
        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        my_dataloader = DataLoader(my_dataset, shuffle=is_shuffle, batch_size=size_batch, num_workers=16
                                   , sampler=sampler)  # ),)  # create your dataloader
        return my_dataloader


def main(dim, train_dataset, test_dataset):
    print(device)

    SEED = 2809
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    ## hyperparams
    num_iterations = 20
    lrmodel = LR(dim)
    lrmodel.to(device)

    loss = torch.nn.MSELoss()  # note that CrossEntropyLoss is for targets with more than 2 classes.
    optimizer = torch.optim.SGD(lrmodel.parameters(), lr=0.0001)

    my_nn = NeuralNetwork(loss_func=loss,
                          optimizer_object=optimizer,
                          model=lrmodel)

    my_nn.fit_model(num_iterations, train_dataset, test_dataset)


if __name__ == "__main__":
    x, y = pr.MainLoader()
    x = x[:100000]
    y = y[:100000]
    DataLoder = DataSet(x, y)
    data_loader, test_loader = DataLoder.split_test_train()
    main(15, data_loader, test_loader)
