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

#==============================================================================

class initialNet(nn.Module):
    def __init__(self, in_dim, hiddens, tau):
        super(initialNet, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [2]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        self.activation = nn.Tanh()
        self.positive = nn.Softplus()
        self.quantile = torch.tensor(norm.ppf(tau), dtype = torch.float32).reshape(1, -1)
    
    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
        X = self.linears[-1](X)
        
        mu = X[:, 0].reshape(-1, 1)
        sig = self.positive(X[:, 1]).reshape(-1, 1)
        Q = torch.matmul(sig, self.quantile) + mu
        return Q

#==============================================================================

class tuningNet(nn.Module):
    def __init__(self, in_dim, hiddens, tau):
        super(tuningNet, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [4]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        self.activation = nn.Tanh()
        self.tau = torch.tensor(tau, dtype = torch.float32).reshape(1, -1)
        self.tau = torch.cat([self.tau ** 0, self.tau, self.tau ** 2, self.tau ** 3], dim = 0)
    
    def forward(self, X, diff, oldQ):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
        coefficents = self.linears[-1](X)
        
        lamb = torch.tanh(torch.matmul(coefficents, self.tau))
        revisions = torch.where(lamb > 0, lamb * diff[:, 1:], lamb * diff[:, :-1])
        newQ = oldQ + revisions
        return newQ, coefficents

#==============================================================================

def cal_diff(Q):
    n, k = Q.shape
    diff = torch.zeros((n, k + 1), requires_grad = False)
    diff[:, 0] = 10
    diff[:, -1] = 10
    diff[:, 1: k] = (Q[:, 1:] - Q[:, :-1]) / 2
    return diff

def pinball_loss(Q, y, tau):
    error = y - Q
    error1 = torch.multiply(error, tau)
    error2 = torch.multiply(error, tau - 1)
    loss = torch.mean(torch.maximum(error1, error2))
    return loss

def penalty_pinball_loss(Q, y, params, C, tau):
    loss1 = pinball_loss(Q, y, tau)
    loss2 = C * torch.mean(params ** 2)
    loss = loss1 + loss2
    return loss

def cal_ECE(true, pred, tau):
    ece = np.mean(np.abs(np.mean(np.array(true).reshape(-1, 1) < pred, axis = 0) - tau.reshape(-1)))
    return ece

#==============================================================================

def load_array(data_arrays, batch_size, is_train = True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle = is_train)

def initialize(model, weight_seed):
    torch.manual_seed(weight_seed)
    for linear in model.linears:
        nn.init.xavier_normal_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

#==============================================================================

def fit_emq(data, var, y, tau, T, C, initial_dims, tuning_dims, num_epochs, batch_size, lr, train_prop, set_seed):
    tau_t = torch.tensor(tau, dtype = torch.float32).reshape(1, -1)
    model = [0] * (T + 1)
    Qlist, Qlist_valid, ECE_valids = [], [], []
    
    X, Y = data[var], data[y]
    X = torch.tensor(X.values, dtype = torch.float32)
    Y = torch.tensor(Y.values, dtype = torch.float32).reshape(-1, 1)
    
    random.seed(set_seed)
    train_index = random.sample(range(len(data)), math.floor(len(data) * train_prop))
    valid_index = list(set(range(len(data))) - set(train_index))
    x_train, y_train, x_valid, y_valid = X[train_index], Y[train_index], X[valid_index], Y[valid_index]
    train_iter = load_array((x_train, y_train), batch_size = batch_size)
    
    model[0] = initialNet(len(var), initial_dims, tau)
    initialize(model[0], set_seed)
    optimizer = optim.Adam(model[0].parameters(), lr = lr)
    
    best_valid = 1e8
    epoch_valids = []
    
    for epoch in range(num_epochs):
        batch_valids = []
        for x_batch, y_batch in train_iter:
            model[0].train()
            loss = pinball_loss(model[0](x_batch), y_batch, tau_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model[0].eval()
            with torch.no_grad():
                valid_loss = pinball_loss(model[0](x_valid), y_valid, tau_t).detach().numpy().item()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model[0].state_dict())
        
        print("model 0, epoch {}:".format(epoch), np.mean(batch_valids))
        epoch_valids.append(np.mean(batch_valids))
        if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) >= np.mean(epoch_valids[-25:-5]):
            break
    
    model[0].load_state_dict(best_dict)
    model[0].eval()
    with torch.no_grad():
        oldQ = model[0](X)
        diff = cal_diff(oldQ)
        Qlist.append(oldQ.detach().numpy())
        
        Qlist_valid.append(model[0](x_valid).detach().numpy())
        ECE_valids.append(cal_ECE(y_valid.detach().numpy(), Qlist_valid[-1], tau))
    
#==============================================================================
    
    for t in range(1, T + 1):
        # random.seed(set_seed + t)
        # train_index = random.sample(range(len(data)), math.floor(len(data) * train_prop))
        # valid_index = list(set(range(len(data))) - set(train_index))
        # x_train, y_train, x_valid, y_valid = X[train_index], Y[train_index], X[valid_index], Y[valid_index]
        
        oldQ_train, oldQ_valid = oldQ[train_index], oldQ[valid_index]
        diff_train, diff_valid = diff[train_index], diff[valid_index]
        train_iter = load_array((x_train, y_train, oldQ_train, diff_train), batch_size = batch_size)
        
        model[t] = tuningNet(len(var), tuning_dims, tau)
        initialize(model[t], set_seed + t)
        nn.init.constant_(model[t].linears[-1].weight, 0.0)
        optimizer = optim.Adam(model[t].parameters(), lr = lr)
        
        model[t].eval()
        with torch.no_grad():
            out, params = model[t](x_valid, diff_valid, oldQ_valid)
            best_valid = pinball_loss(out, y_valid, tau_t).detach().numpy().item()
            best_dict = copy.deepcopy(model[t].state_dict())
        epoch_valids = []
        
        for epoch in range(num_epochs):
            batch_valids = []
            for x_batch, y_batch, oldQ_batch, diff_batch in train_iter:
                model[t].train()
                out, params = model[t](x_batch, diff_batch, oldQ_batch)
                loss = penalty_pinball_loss(out, y_batch, params, C, tau_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                model[t].eval()
                with torch.no_grad():
                    out, params = model[t](x_valid, diff_valid, oldQ_valid)
                    valid_loss = pinball_loss(out, y_valid, tau_t).detach().numpy().item()
                batch_valids.append(valid_loss)
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    best_dict = copy.deepcopy(model[t].state_dict())
            
            print("model {}, epoch {}:".format(t, epoch), np.mean(batch_valids))
            epoch_valids.append(np.mean(batch_valids))
            if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) >= np.mean(epoch_valids[-25:-5]):
                break
        
        model[t].load_state_dict(best_dict)
        model[t].eval()
        with torch.no_grad():
            oldQ, params = model[t](X, diff, oldQ)
            diff = cal_diff(oldQ)
            Qlist.append(oldQ.detach().numpy())
            
            oldQ_valid, params = model[t](x_valid, diff_valid, oldQ_valid)
            Qlist_valid.append(oldQ_valid.detach().numpy())
            ECE_valids.append(cal_ECE(y_valid.detach().numpy(), Qlist_valid[-1], tau))
            if len(ECE_valids) >= 10 and np.mean(ECE_valids[-2:]) > np.mean(ECE_valids[-10:-2]):
                break
    
    Tnew = np.argmin(ECE_valids)
    return model[:Tnew + 1], Qlist[:Tnew + 1]

#==============================================================================

def emq_predict(model, test_data, var):
    X_test = test_data[var]
    X_test = torch.tensor(X_test.values, dtype = torch.float32)
    
    Qlist = []
    with torch.no_grad():
        oldQ = model[0](X_test)
        diff = cal_diff(oldQ)
        Qlist.append(oldQ.detach().numpy())
        
        for t in range(1, len(model)):
            oldQ, params = model[t](X_test, diff, oldQ)
            diff = cal_diff(oldQ)
            Qlist.append(oldQ.detach().numpy())
    
    Q_predict = oldQ.detach().numpy()
    return Q_predict, Qlist