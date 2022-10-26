# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

#==============================================================================

tau = np.array(range(1, 100)) / 100

def gen_quantiles(mu, variance, tau):
    quantiles = np.dot(np.sqrt(variance).reshape(-1, 1), norm.ppf(tau).reshape(1, -1))
    quantiles = quantiles + np.array(mu).reshape(-1, 1)
    return quantiles

def cal_ECE(true, pred, tau):
    ece = np.mean(np.abs(np.mean(np.array(true).reshape(-1, 1) < pred, axis = 0) - tau.reshape(-1)))
    return ece

def cal_interval_ECE(true, pred, tau):
    true_s = np.array(true).reshape(-1,1)
    compare = np.logical_and(true_s > pred[:,:49], true_s < pred[:,-49:][:,::-1])
    ece = np.mean(np.abs(np.mean(compare, axis = 0) - (1-tau.reshape(-1)[:49]*2)))
    return ece

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

method = 'emqw'

#==============================================================================

def capitalize(data_name):
    if data_name == 'pm2.5':
        return "PM2.5"
    elif data_name == "gpu":
        return "GPU"
    else:
        return data_name.capitalize()

linestyles = ['solid','dotted','dashed','dashdot','solid','dotted','dashed',
              'dashdot','solid','dotted','dashed','dashdot']
markers = ['o','v','^','s','*','+','x','D','1','X','d','2']
colors = ['b','g','r','c','m','y','k','gray','pink','lime','lightblue','orange']

random.seed(10)
for data_name in data_list:
    resultDir = "../../{}/results/{}/{}/".format(method,data_name,method)
    
    fileList = pd.DataFrame(os.listdir(resultDir),columns=['fileName'])
    fileList['seed'] = fileList['fileName'].apply(lambda x:int(x.split('_')[1]))
    fileList['number'] = fileList['fileName'].apply(lambda x:int(x.split('_')[-1][:-4]))
    fileList.sort_values(['seed','number'],inplace=True)
    
    fileName0 = fileList.groupby('seed',sort=False).first()['fileName'].tolist()[0]
    fileName = fileList.groupby('seed',sort=False).last()['fileName'].tolist()[0]
    Tada = int(fileName.split('_')[-1][:-4])
    results0 = pd.read_csv(resultDir+fileName0)
    results = pd.read_csv(resultDir+fileName)
    quantiles0 = results0[results0.columns[:-1]].values
    quantiles = results[results.columns[:-1]].values
    
    index = {'air':[137053,147867],'blog':[7847,4740],'election':[663,1226],
             'facebook2':[9549,15083],'naval':[16,1725],'gpu':[22806,22382],
             'video':[13498,10274],'consumption':[4356,9750],'query':[38782,35376]}
    if data_name not in index:
        continue
    i = 0
    for ind in index[data_name]:
        i = i+1
        pdf = PdfPages("density/{}-{}.pdf".format(data_name,i))
        plt.figure(figsize=(5.7,3.8),dpi=300)
        
        xs = quantiles0[ind][1:-1]
        ys = 0.02/(quantiles0[ind][2:]-quantiles0[ind][:-2])
        plt.plot(xs, ys, linestyle='dotted', color='r', label='$t = 0$', linewidth=3)
        
        xs = quantiles[ind][1:-1]
        ys = 0.02/(quantiles[ind][2:]-quantiles[ind][:-2])
        plt.plot(xs, ys, linestyle='solid', color='b', label='$t = T_{{ada}} = {}$'.format(Tada), linewidth=3)
        
        plt.legend(loc='best', fontsize=20)
        plt.title(capitalize(data_name), fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        pdf.savefig()
        plt.close()
        pdf.close()

with open("density/tex.txt", 'w') as file:
    j = 1
    for data_name in ['consumption','gpu','air','election','query','facebook2','blog','video','naval']:
        for i in [1,2]:
            file.write("\\begin{minipage}[t]{.16\\linewidth}\n")
            file.write("\\centerline{{\\includegraphics[width=\\linewidth]{{plots/density/{}-{}.pdf}}}}\n".format(data_name,i))
            if j % 6 == 0:
                file.write("\\end{minipage}\\\\\n")
            else:
                file.write("\\end{minipage}\\hfill\n")
            j = j+1



