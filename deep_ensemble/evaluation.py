# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from fit_hnn import *

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

    method_name = 'single_ensemble'
    
    num_epochs = 1000
    batch_size = 128
    lr = 0.01
    train_prop = 0.8
    num_iter = 2
    
    dims = [8, 4]
    in_dim = len(var)
    hidden_dims = list(np.array(dims) * in_dim)
    
    model_mu, model_var, _, _ = fit_hnn(train_data, y, num_iter, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed)
    
    test_input = torch.tensor(test_data[var].values, dtype = torch.float32)
    with torch.no_grad():
        mu_hnn = model_mu(test_input).detach().numpy().squeeze()
        var_hnn = model_var(test_input).detach().numpy().squeeze()
    
    quantiles = gen_quantiles(mu_hnn, var_hnn, tau)
    
    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,hidden_dims,ECE,rmse\n')
    with open(data_dir + method_name + ".csv", 'a') as file:
        file.write("{},\"{}\",{},{}\n".format(set_seed, str(dims), cal_ECE(test_data[y], quantiles, tau), rmse(mu_hnn, test_data[y])))
    
    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results = pd.DataFrame([mu_hnn.tolist(), var_hnn.tolist(), test_data[y].tolist()]).T
    results.columns = ['mean', 'variance', 'true']
    results.to_csv(data_dir + method_name + "/" + "hnn_{}_{}.csv".format(set_seed, str(dims)), index = False)
    
#==============================================================================
    
    method_name = 'hnn_ensemble'
    
    num_epochs = 1000
    batch_size = 128
    lr = 0.01
    train_prop = 0.8
    num_iter = 2
    nn_Num = 5
    
    dims = [8, 4]
    in_dim = len(var)
    hidden_dims = list(np.array(dims) * in_dim)
    
    mu_ensem = np.zeros((len(test_data), nn_Num))
    var_ensem = np.zeros((len(test_data), nn_Num))
    mu_ensem[:, 0] = mu_hnn
    var_ensem[:, 0] = var_hnn
    
    for n in range(1, nn_Num):
        model_mu, model_var, _, _ = fit_hnn(train_data, y, num_iter, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed + n * 1997)
        
        with torch.no_grad():
            mu_ensem[:, n] = model_mu(test_input).detach().numpy().squeeze()
            var_ensem[:, n] = model_var(test_input).detach().numpy().squeeze()
    
    var_ensem2 = var_ensem.mean(axis = 1)
    var_ensem = (var_ensem + mu_ensem ** 2).mean(axis = 1) - mu_ensem.mean(axis = 1) ** 2
    mu_ensem = mu_ensem.mean(axis = 1)
    quantiles = gen_quantiles(mu_ensem, var_ensem, tau)
    quantiles2 = gen_quantiles(mu_ensem, var_ensem2, tau)
    
    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,hidden_dims,nn_Num,ECE,ECE2,rmse\n')
    with open(data_dir + "hnn_ensemble.csv", 'a') as file:
        file.write("{},\"{}\",{},{},{},{}\n".format(set_seed, str(dims), nn_Num, cal_ECE(test_data[y], quantiles, tau), 
                                                    cal_ECE(test_data[y], quantiles2, tau), rmse(mu_ensem, test_data[y])))
    
    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results = pd.DataFrame([mu_ensem.tolist(), var_ensem.tolist(), var_ensem2.tolist(), test_data[y].tolist()]).T
    results.columns = ['mean', 'variance', 'variance2', 'true']
    results.to_csv(data_dir + method_name + "/" + "hnn_ensemble_{}_{}_{}.csv".format(set_seed, str(dims), nn_Num), index = False)

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