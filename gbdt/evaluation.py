# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss

from sklearn.ensemble import GradientBoostingRegressor

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
    
    method_name = 'gbdt'
    
    train_prop = 0.8
    random.seed(set_seed)
    train_index = random.sample(range(len(train_data)), math.floor(len(train_data) * train_prop))
    valid_index = list(set(range(len(train_data))) - set(train_index))
    train_data_s, valid_data = train_data.iloc[train_index], train_data.iloc[valid_index]
    
    quantiles = np.zeros((test_data.shape[0],99))
    
    for level in tau:
        best_valid_loss = 1e8
        for max_depth in [3, 4, 5]:
            for n_estimators in [50, 100, 200]:
                
                gbdt = GradientBoostingRegressor(loss = 'quantile', alpha = level, max_depth = max_depth,
                                                 n_estimators = n_estimators, random_state = set_seed)
                gbdt.fit(train_data_s[var], train_data_s[y])
                valid_loss = mean_pinball_loss(valid_data[y], gbdt.predict(valid_data[var]), alpha = level)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_gbdt = {'max_depth':max_depth, 'n_estimators':n_estimators}
        
        gbdt = GradientBoostingRegressor(loss = 'quantile', alpha = level, max_depth = best_gbdt['max_depth'],
                                         n_estimators = best_gbdt['n_estimators'], random_state = set_seed)
        gbdt.fit(train_data[var], train_data[y])
        quantiles[:,int(level*100-1)] = gbdt.predict(test_data[var])
   
    pred = np.mean(quantiles, axis = 1)
    
    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,ECE,rmse\n')
    with open(data_dir + method_name + ".csv", 'a') as file:
        file.write("{},{},{}\n".format(set_seed, cal_ECE(test_data[y], quantiles, tau), rmse(pred, test_data[y])))
    
    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results = pd.DataFrame(quantiles, columns = np.array(range(1,100))/100)
    results['true'] = test_data[y].tolist()
    results.to_csv(data_dir + method_name + "/{}_{}.csv".format(method_name, set_seed), index = False)

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