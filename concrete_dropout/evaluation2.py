# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from concrete_dropout import *

def rmse(x, y):
    x_ = np.array(x).reshape(-1)
    y_ = np.array(y).reshape(-1)
    return np.sqrt(((x_ - y_) ** 2).mean())

#==============================================================================

def gen_quantiles(mu, variance, tau):
    quantiles = np.dot(np.sqrt(variance).reshape(-1, 1), norm.ppf(tau).reshape(1, -1))
    quantiles = quantiles + np.array(mu).reshape(-1, 1)
    return quantiles

def cal_ECE(true, pred, tau):
    ece = np.mean(np.abs(np.mean(np.array(true).reshape(-1, 1) < pred, axis = 0) - tau.reshape(-1)))
    return ece

#==============================================================================

def calibration(data_name, train_data, test_data, var, y, set_seed):
    if not os.path.isdir("results/"+data_name):
        os.mkdir("results/"+data_name)
    data_dir = "results/"+data_name + '/'
    
    tau = np.array(range(1, 100)) / 100
    
#==============================================================================
    
    method_name = 'concrete_dropout'
    
    num_epochs = 1000
    batch_size = 128
    lr = 0.01
    train_prop = 0.8
    
    dims = [8, 4]
    in_dim = len(var)
    hidden_dims = [100, 80]#list(np.array(dims) * in_dim)
    
    model_cdp = fit_cdp(train_data[var], train_data[y], hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed)
    model_cdp.eval()
    
    num_cdp = 1000
    pred_mu = np.zeros((test_data.shape[0], num_cdp))
    pred_logvar = np.zeros((test_data.shape[0], num_cdp))
    test_input = torch.tensor(test_data[var].values, dtype = torch.float32)
    for t in range(num_cdp):
        mu, logvar, reg = model_cdp(test_input)
        pred_mu[:, t], pred_logvar[:, t] = mu.detach().numpy().squeeze(), logvar.detach().numpy().squeeze()
    
    mu_cdp = np.mean(pred_mu, axis=1)
    var_cdp = np.mean(np.exp(pred_logvar), axis=1)
    quantiles = gen_quantiles(mu_cdp, var_cdp, tau)

    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,ECE,rmse\n')
    with open(data_dir + method_name + ".csv", 'a') as file:
        file.write("{},{},{}\n".format(set_seed, cal_ECE(test_data[y], quantiles, tau), rmse(mu_cdp, test_data[y])))

    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results = pd.DataFrame([mu_cdp.tolist(), var_cdp.tolist(), test_data[y].tolist()]).T
    results.columns = ['mean', 'variance', 'true']
    results.to_csv(data_dir + method_name + "/" + "concrete_dropout_{}.csv".format(set_seed), index = False)

#==============================================================================

def aver_calibrate(data_name, data, var, y):
    if not os.path.isdir("results/"):
        os.mkdir("results/")
    print('')
    print(data_name)
    print(len(data), len(var))
    
    N_aver = 5 if len(data) < 1e5 else 1
    Train_proportion = 0.8 if len(data) < 3e5 else 0.5
    
    data_new = pd.DataFrame(data[var], copy = True)
    data_new[y] = data[y]
    data_new = (data_new - data_new.mean()) / data_new.std()
    
    for i in range(N_aver):
        set_seed = 42 + 100 * i
        
        random.seed(set_seed)
        train_index = random.sample(range(len(data)), math.floor(len(data) * Train_proportion))
        test_index = list(set(range(len(data))) - set(train_index))
        
        train_data = data_new.iloc[train_index]
        test_data = data_new.iloc[test_index]
        calibration(data_name, train_data, test_data, var, y, set_seed)