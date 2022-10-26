# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from emqw import *

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
# emqw
    
    num_epochs = 1000
    batch_size = 128
    lr = 0.01
    train_prop = 0.8
    
    T = 40
    C = 0
    
    in_dim = len(var)
    dims_i = [8, 16, 4]
    initial_dims = list(np.array(dims_i) * in_dim)
    dims_t = [16, 8]#[4, 2]
    tuning_dims = dims_t#list(np.array(dims_t) * in_dim)
    
    model, Qlist = fit_emq(train_data, var, y, tau, T, C, initial_dims, tuning_dims, num_epochs, batch_size, lr, train_prop, set_seed)
    Q_predict, Qlist = emq_predict(model, test_data, var)
    
    if not os.path.isfile(data_dir + "emqw.csv"):
        with open(data_dir + "emqw.csv", 'a') as file:
            file.write('train_test_seed,T,C,initial_dims,tuning_dims,ECE')
            for t in range(len(Qlist) - 1):
                file.write(',ECE'+str(t))
            file.write('\n')
    with open(data_dir + "emqw.csv", 'a') as file:
        file.write("{},{},{},\"{}\",\"{}\",{}".format(set_seed, T, C, str(dims_i), str(dims_t), cal_ECE(test_data[y], Q_predict, tau)))
        for t in range(len(Qlist) - 1):
            file.write(",{}".format(cal_ECE(test_data[y], Qlist[t], tau)))
        file.write('\n')
    
    if not os.path.isdir(data_dir + "emqw/"):
        os.mkdir(data_dir + "emqw/")
    for t in range(len(Qlist)):
        results = pd.DataFrame(Qlist[t], columns = tau)
        results['true'] = test_data[y].tolist()
        results.to_csv(data_dir + "emqw/" + "emqw_{}_{}_{}_{}_{}_{}.csv".format(set_seed, T, C, str(dims_i), str(dims_t), t), index = False)

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