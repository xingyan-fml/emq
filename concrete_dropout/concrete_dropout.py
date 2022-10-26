# -*- coding: utf-8 -*-

import copy
import math
import random
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

#==============================================================================

def load_array(data_arrays, batch_size, is_train = True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle = is_train)

def initialize(model, weight_seed):
    torch.manual_seed(weight_seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

#==============================================================================

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()

        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        
    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        
        out = layer(self._concrete_dropout(x, p))
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
        
    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return x

#==============================================================================

class Model(nn.Module):
    def __init__(self, indim, hidden_dims, weight_regularizer, dropout_regularizer):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(indim, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.linear3_mu = nn.Linear(hidden_dims[1], 1)
        self.linear3_logvar = nn.Linear(hidden_dims[1], 1)

        self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu = ConcreteDropout(weight_regularizer=weight_regularizer,
                                             dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = ConcreteDropout(weight_regularizer=weight_regularizer,
                                                 dropout_regularizer=dropout_regularizer)
        
        # self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
    def forward(self, x):
        regularization = torch.empty(4, device=x.device)
        
        x1, regularization[0] = self.conc_drop1(x, nn.Sequential(self.linear1, self.elu))
        x2, regularization[1] = self.conc_drop2(x1, nn.Sequential(self.linear2, self.elu))

        mean, regularization[2] = self.conc_drop_mu(x2, self.linear3_mu)
        log_var, regularization[3] = self.conc_drop_logvar(x2, self.linear3_logvar)

        return mean, log_var, regularization.sum()

def heteroscedastic_loss(true, mean, log_var):
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(precision * (true - mean)**2 + log_var, 1), 0)

#==============================================================================

D = 1 # One mean, one log_var
l = 1e-4 # Lengthscale

#==============================================================================

def fit_cdp(X, Y, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed):
    
    random.seed(set_seed)
    train_index = random.sample(range(len(X)), math.floor(len(X) * train_prop))
    valid_index = list(set(range(len(X))) - set(train_index))
    
    x_train, y_train = X.iloc[train_index], Y.iloc[train_index]
    x_valid, y_valid = X.iloc[valid_index], Y.iloc[valid_index]
    
    x_train = torch.tensor(x_train.values, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32).reshape(-1, 1)
    x_valid = torch.tensor(x_valid.values, dtype = torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype = torch.float32).reshape(-1, 1)
    
    N = x_train.shape[0]
    wr = l**2. / N
    dr = 2. / N
    model = Model(x_train.shape[1], hidden_dims, wr, dr)
    
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
            mean, log_var, regularization = model(x_batch)
            loss = heteroscedastic_loss(y_batch, mean, log_var) + regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            mean_val, log_var_val, regularization_val = model(x_valid)
            valid_loss = heteroscedastic_loss(y_valid, mean_val, log_var_val).detach().numpy()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model.state_dict())
        
        batch_valids = np.mean(batch_valids)
        print("epoch {}:".format(epoch), batch_valids)
        if np.isnan(batch_valids):
            break
        epoch_valids.append(batch_valids)
        if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) > np.mean(epoch_valids[-25:-5]):
            break
    
    model.load_state_dict(best_dict)
    model.eval()
    
    return model


