# -*- coding: utf-8 -*-

import copy
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


# ==============================================================================

class vanillaNet(nn.Module):
    def __init__(self, in_dim, hiddens):
        super(vanillaNet, self).__init__()

        self.dims = [in_dim] + list(hiddens) + [99]

        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i - 1], self.dims[i]))

        self.activation = nn.ReLU()

    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
        Q = self.linears[-1](X)
        return Q

# ==============================================================================


def pinball_loss(Q, y, tau):
    error = y - Q
    error1 = torch.multiply(error, tau)
    error2 = torch.multiply(error, tau - 1)
    loss = torch.mean(torch.maximum(error1, error2))
    return loss


# ==============================================================================

def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


def initialize(model, weight_seed):
    torch.manual_seed(weight_seed)
    for linear in model.linears:
        nn.init.xavier_normal_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)


# ==============================================================================

def fit_vanilla(data, var, y, tau, initial_dims, num_epochs, batch_size, lr, train_prop, set_seed):
    tau = torch.tensor(tau, dtype = torch.float32).reshape(1, -1)
    X, Y = data[var], data[y]
    X = torch.tensor(X.values, dtype=torch.float32)
    Y = torch.tensor(Y.values, dtype=torch.float32).reshape(-1, 1)

    random.seed(set_seed)
    train_index = random.sample(range(len(data)), math.floor(len(data) * train_prop))
    valid_index = list(set(range(len(data))) - set(train_index))
    x_train, y_train, x_valid, y_valid = X[train_index], Y[train_index], X[valid_index], Y[valid_index]
    train_iter = load_array((x_train, y_train), batch_size=batch_size)

    model = vanillaNet(len(var), initial_dims)
    initialize(model, set_seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valid = 1e8
    epoch_valids = []

    for epoch in range(num_epochs):
        batch_valids = []
        for x_batch, y_batch in train_iter:
            model.train()
            loss = pinball_loss(model(x_batch), y_batch, tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_loss = pinball_loss(model(x_valid), y_valid, tau).detach().numpy().item()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model.state_dict())

        print("vanilla, epoch {}:".format(epoch), np.mean(batch_valids))
        epoch_valids.append(np.mean(batch_valids))
        if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) > np.mean(epoch_valids[-25:-5]):
            break

    model.load_state_dict(best_dict)
    model.eval()
    return model


# ==============================================================================

def vanilla_predict(model, test_data, var):
    X_test = test_data[var]
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        Q_predict = model(X_test).detach().numpy()
    return Q_predict