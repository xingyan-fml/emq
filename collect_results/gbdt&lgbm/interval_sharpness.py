# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

#==============================================================================

tau = np.array(range(1, 100)) / 100

def gen_quantiles(mu, variance, tau):
    quantiles = np.dot(np.sqrt(variance).reshape(-1, 1), norm.ppf(tau).reshape(1, -1))
    quantiles = quantiles + np.array(mu).reshape(-1, 1)
    return quantiles

def cal_sharpness(true, pred, tau):
    diff = pred[:,-49:][:,::-1] - pred[:,:49]
    sharpness = np.mean(np.mean(diff,axis=0))
    return sharpness

#==============================================================================

data_list = ['grid',
'naval',
'appliance',
'election',
'steel',
'facebook1',
'facebook2',
'pm2.5',
'bio',
'blog',
'consumption',
'video',
'gpu',
'query',
'wave',
'air',
'year',
]

methods = ['emqw','gbdt','lgbm']

methods_disp = ['EMQW','GBDT','LightGBM']
methods_disp = dict(zip(methods,methods_disp))

#==============================================================================

all_ece = []

for data_name in data_list:
    ece_methods = []
    ece = []
    
    for method in methods:
        if method == "emq0":
            resultDir = "../../{}/results/{}/{}/".format('emq',data_name,'emq')
        elif method in ['single_ensemble','hnn_ensemble']:
            resultDir = "../../deep_ensemble/results/{}/{}/".format(data_name,method)
        else:
            resultDir = "../../{}/results/{}/{}/".format(method,data_name,method)
        
        if "emq" in method:
            fileList = pd.DataFrame(os.listdir(resultDir),columns=['fileName'])
            fileList['seed'] = fileList['fileName'].apply(lambda x:int(x.split('_')[1]))
            fileList['number'] = fileList['fileName'].apply(lambda x:int(x.split('_')[-1][:-4]))
            fileList.sort_values(['seed','number'],inplace=True)
            if method == "emq0":
                fileList = fileList.groupby('seed',sort=False).first()['fileName'].tolist()
            else:
                fileList = fileList.groupby('seed',sort=False).last()['fileName'].tolist()
        else:
            fileList = os.listdir(resultDir)
        
        sharpness = []
        for fileName in fileList:
            results = pd.read_csv(resultDir+fileName)
            if len(results.columns) < 100:
                quantiles = gen_quantiles(results['mean'].values, results['variance'].values, tau)
            else:
                quantiles = results[results.columns[:-1]].values
            sharpness.append(cal_sharpness(results['true'], quantiles, tau))
        sharpness = np.mean(sharpness)
        
        ece_methods.append(method)
        ece.append(sharpness)
    
    all_ece.append(pd.DataFrame([ece], columns = ece_methods))

all_ece = pd.concat(all_ece, ignore_index = True)
all_ece.index = data_list

#==============================================================================

def capitalize(data_name):
    if data_name == 'pm2.5':
        return "PM2.5"
    elif data_name == "gpu":
        return "GPU"
    else:
        return data_name.capitalize()

with open("interval_sharpness.txt", 'w') as file:
    file.write("Dataset $\\backslash$ Method")
    for method in all_ece.columns:
        file.write(" & "+methods_disp[method])
    file.write(" \\\\ \\Xhline{1pt}\n")
    for data_name in data_list:
        ece = all_ece.loc[data_name]
        idxmin = ece.sort_values().index[0]
        # if idxmin == 'mc_dropout':
        #     idxmin = ece.sort_values().index[1]
        #     idxmin2 = ece.sort_values().index[2]
        # else:
        #     idxmin2 = ece.sort_values().index[1]
        idxmin2 = ece.sort_values().index[1]
        file.write(capitalize(data_name))
        for method in ece.index:
            if method == idxmin:
                file.write(" & $^{{**}}${:.2f}".format(ece[method]*100))
            elif method == idxmin2:
                file.write(" & $^*${:.2f}".format(ece[method]*100))
            else:
                file.write(" & {:.2f}".format(ece[method]*100))
        if data_name == data_list[-1]:
            file.write(" \\\\ \\Xhline{1pt}\n")
        else:
            file.write(" \\\\ \\hline\n")

for data_name in data_list:
    ece = all_ece.loc[data_name]
    idxmin = ece.sort_values().index[0]
    all_ece.loc[data_name, idxmin] = '*' + str(all_ece.loc[data_name, idxmin])

all_ece.to_csv("interval_sharpness.csv")



