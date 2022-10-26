# -*- coding: utf-8 -*-

import os
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

for data_name in data_list:
    resultDir = "../../{}/results/{}/{}/".format(method,data_name,method)
    
    fileList = pd.DataFrame(os.listdir(resultDir),columns=['fileName'])
    fileList['seed'] = fileList['fileName'].apply(lambda x:int(x.split('_')[1]))
    fileList['number'] = fileList['fileName'].apply(lambda x:int(x.split('_')[-1][:-4]))
    fileList.sort_values(['seed','number'],inplace=True)
    
    interval_ECE = {}
    for seed, seedFiles in fileList.groupby('seed',sort=False):
        interval_ECE[seed] = []
        for fileName in seedFiles['fileName'].tolist():
            results = pd.read_csv(resultDir+fileName)
            quantiles = results[results.columns[:-1]].values
            interval_ECE[seed].append(cal_interval_ECE(results['true'], quantiles, tau)*100)
    
    pdf = PdfPages("Tada/{}.pdf".format(data_name))
    plt.figure(figsize=(5.7,3.8),dpi=300)
    i = 0
    length = []
    for seed in sorted(interval_ECE.keys()):
        plt.plot(interval_ECE[seed], linestyle=linestyles[i], marker=markers[i], color=colors[i],
                 markersize=10, linewidth=3)
        i = i+1
        length.append(len(interval_ECE[seed])-1)
    
    # plt.legend(title="{}, average $T_{{ada}}$ = {:.1f}".format(capitalize(data_name),np.mean(length)),
    #            fontsize='large', loc='upper right')
    plt.title("{}, average $T_{{ada}}$ = {:.1f}".format(capitalize(data_name),np.mean(length)), fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    pdf.savefig()
    plt.close()
    pdf.close()

with open("Tada/tex.txt", 'w') as file:
    for data_name in data_list:
        file.write("\\begin{minipage}[t]{.165\\linewidth}\n")
        file.write("\\centerline{{\\includegraphics[width=\\linewidth]{{plots/Tada/{}.pdf}}}}\n".format(data_name))
        file.write("\\end{minipage}\n")



