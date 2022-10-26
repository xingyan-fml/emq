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

class model_nn(nn.Module):
    def __init__(self, in_dim, hiddens):
        super(model_nn, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [1]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        self.activation = nn.ReLU()
    
    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
        X = self.linears[-1](X)
        return X

class variance_nn(nn.Module):
    def __init__(self, in_dim, hiddens):
        super(variance_nn, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [1]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        self.activation = nn.Tanh()
        self.positive = nn.Softplus()
    
    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
        X = self.linears[-1](X)
        X = self.positive(X) + 1e-8
        return X

#==============================================================================

# Negative log-likelihood loss function
def NLLloss(y, mean, var):
    return (torch.log(var) + (y - mean).pow(2) / var).sum()

def MSELoss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def load_array(data_arrays, batch_size, is_train = True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle = is_train)

def initialize(model, weight_seed):
    torch.manual_seed(weight_seed)
    for linear in model.linears:
        nn.init.xavier_normal_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

#==============================================================================

def fit_weight_nn(data, y, v_hat, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed):
    model = model_nn(data.shape[1] - 1, hidden_dims)
    
    random.seed(set_seed)
    train_index = random.sample(range(len(data)), math.floor(len(data) * train_prop))
    valid_index = list(set(range(len(data))) - set(train_index))
    
    X, Y = data.drop(y, axis = 1), data[y]
    x_train, y_train, v_hat_train = X.iloc[train_index], Y.iloc[train_index], v_hat[train_index]
    x_valid, y_valid, v_hat_valid = X.iloc[valid_index], Y.iloc[valid_index], v_hat[valid_index]
    
    X = torch.tensor(X.values, dtype = torch.float32)
    Y = torch.tensor(Y.values, dtype = torch.float32).reshape(-1, 1)
    x_train = torch.tensor(x_train.values, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32).reshape(-1, 1)
    x_valid = torch.tensor(x_valid.values, dtype = torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype = torch.float32).reshape(-1, 1)
    weight_train = 1 / torch.tensor(v_hat_train, dtype = torch.float32).reshape(-1, 1)
    weight_valid = 1 / torch.tensor(v_hat_valid, dtype = torch.float32).reshape(-1, 1)
    
    initialize(model, set_seed)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    if batch_size < len(x_train):
        train_iter = load_array((x_train, y_train, weight_train), batch_size = batch_size)
    else:
        train_iter = [(x_train, y_train, weight_train)]
    
    best_valid = 1e8
    epoch_valids = []
    
    for epoch in range(num_epochs):
        batch_valids = []
        for x_batch, y_batch, weight_batch in train_iter:
            model.train()
            out = model(x_batch)
            loss = MSELoss(out, y_batch, weight_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                valid_loss = MSELoss(model(x_valid), y_valid, weight_valid).detach().numpy()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model.state_dict())
        
        print("epoch {}:".format(epoch), np.mean(batch_valids))
        epoch_valids.append(np.mean(batch_valids))
        if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) >= np.mean(epoch_valids[-25:-5]):
            break
    
    model.load_state_dict(best_dict)
    model.eval()
    valid_loss = MSELoss(model(x_valid), y_valid, weight_valid).detach().numpy()
    print("final:", valid_loss, best_valid)
    Y_hat = model(X).detach().numpy().reshape(-1)
    
    return model, Y_hat

#==============================================================================

def fit_variance_nn(data, y, Y_hat, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed):
    model = variance_nn(data.shape[1] - 1, hidden_dims)
    
    random.seed(set_seed)
    train_index = random.sample(range(len(data)), math.floor(len(data) * train_prop))
    valid_index = list(set(range(len(data))) - set(train_index))
    
    X, Y = data.drop(y, axis = 1), data[y]
    x_train, y_train, y_hat_train = X.iloc[train_index], Y.iloc[train_index], Y_hat[train_index]
    x_valid, y_valid, y_hat_valid = X.iloc[valid_index], Y.iloc[valid_index], Y_hat[valid_index]
    
    X = torch.tensor(X.values, dtype = torch.float32)
    Y = torch.tensor(Y.values, dtype = torch.float32).reshape(-1, 1)
    x_train = torch.tensor(x_train.values, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32).reshape(-1, 1)
    x_valid = torch.tensor(x_valid.values, dtype = torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype = torch.float32).reshape(-1, 1)
    y_hat_train = torch.tensor(y_hat_train, dtype = torch.float32).reshape(-1, 1)
    y_hat_valid = torch.tensor(y_hat_valid, dtype = torch.float32).reshape(-1, 1)
    
    initialize(model, set_seed)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    if batch_size < len(x_train):
        train_iter = load_array((x_train, y_train, y_hat_train), batch_size = batch_size)
    else:
        train_iter = [(x_train, y_train, y_hat_train)]
    
    best_valid = 1e8
    epoch_valids = []
    
    for epoch in range(num_epochs):
        batch_valids = []
        for x_batch, y_batch, y_hat_batch in train_iter:
            model.train()
            out = model(x_batch)
            loss = NLLloss(y_batch, y_hat_batch, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                valid_loss = NLLloss(y_valid, y_hat_valid, model(x_valid)).detach().numpy()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model.state_dict())
        
        print("epoch {}:".format(epoch), np.mean(batch_valids))
        epoch_valids.append(np.mean(batch_valids))
        if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) >= np.mean(epoch_valids[-25:-5]):
            break
    
    model.load_state_dict(best_dict)
    model.eval()
    valid_loss = NLLloss(y_valid, y_hat_valid, model(x_valid)).detach().numpy()
    print("final:", valid_loss, best_valid)
    var_hat = model(X).detach().numpy().reshape(-1)
    
    return model, var_hat

#==============================================================================

def fit_hnn(data, y, num_iter, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed):
    v_hat = np.ones(len(data))
    
    for n in range(num_iter):
        model_mu, Y_hat = fit_weight_nn(data, y, v_hat, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed + n * 101)
        model_var, v_hat = fit_variance_nn(data, y, Y_hat, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed + n * 101 + 2)
    
    return model_mu, model_var, Y_hat, v_hat

#==============================================================================

if __name__ == "__main__":
    
    N = 10000
    np.random.seed(12)
    data_x = np.random.randn(N, 3)
    mean_x = np.dot(data_x, np.array([[0.1], [0.3], [-0.2]])) + 0.5
    std_x = np.sqrt(np.dot(data_x ** 2, np.array([[0.02], [0.03], [0.04]])))
    data_y = mean_x + std_x * np.random.randn(N, 1)
    
    data = pd.DataFrame(data_x, columns = range(data_x.shape[1]))
    data['y'] = data_y
    
    hidden_dims = [32, 8]
    model_mu, model_var, Y_hat, v_hat = fit_hnn(data, data.columns[:-1], [], 'y', 4, hidden_dims,
                num_epochs=100, batch_size=64, lr=0.01, train_prop=0.8, set_seed=10)
    
    compare = pd.DataFrame(columns = ['mean', 'pred_mean', 'std', 'pred_std'])
    compare['mean'] = mean_x.reshape(-1)
    compare['pred_mean'] = Y_hat
    compare['std'] = std_x.reshape(-1)
    compare['pred_std'] = np.sqrt(v_hat)
    
    print(np.corrcoef(compare['mean'], compare['pred_mean']))
    print(np.corrcoef(compare['std'], compare['pred_std']))


