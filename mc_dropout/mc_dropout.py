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

class model_mcdropout(nn.Module):
    def __init__(self, in_dim, hiddens, dropout):
        super(model_mcdropout, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [1]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        self.dropout = nn.Dropout(p = dropout)
        self.activation = nn.ReLU()
    
    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
            X = self.dropout(X)
        X = self.linears[-1](X)
        return X

#==============================================================================

criterion = nn.MSELoss()

def load_array(data_arrays, batch_size, is_train = True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle = is_train)

def initialize(model, weight_seed):
    torch.manual_seed(weight_seed)
    for linear in model.linears:
        nn.init.xavier_normal_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

#==============================================================================

def fit_mcdropout(X, Y, hidden_dims, dropout, num_epochs, batch_size, lr, train_prop, set_seed):
    model = model_mcdropout(X.shape[1], hidden_dims, dropout)
    
    random.seed(set_seed)
    train_index = random.sample(range(len(X)), math.floor(len(X) * train_prop))
    valid_index = list(set(range(len(X))) - set(train_index))
    
    x_train, y_train = X.iloc[train_index], Y.iloc[train_index]
    x_valid, y_valid = X.iloc[valid_index], Y.iloc[valid_index]
    
    x_train = torch.tensor(x_train.values, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32).reshape(-1, 1)
    x_valid = torch.tensor(x_valid.values, dtype = torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype = torch.float32).reshape(-1, 1)
    
    initialize(model, set_seed)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    if batch_size < len(x_train):
        train_iter = load_array((x_train, y_train), batch_size = batch_size)
    else:
        train_iter = [(x_train, y_train)]
    
    best_valid = 1e8
    epoch_valids = []
    
    for epoch in range(num_epochs):
        batch_valids = []
        for x_batch, y_batch in train_iter:
            model.train()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                valid_loss = criterion(model(x_valid), y_valid).detach().numpy()
                batch_valids.append(valid_loss)
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    best_dict = copy.deepcopy(model.state_dict())
        
        print("epoch {}:".format(epoch), np.mean(batch_valids))
        epoch_valids.append(np.mean(batch_valids))
        if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) > np.mean(epoch_valids[-25:-5]):
            break
    
    model.load_state_dict(best_dict)
    model.train()
    
    return model


