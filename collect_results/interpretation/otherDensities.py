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

methods = ['vanilla','vanilla.w','sqr','interval']
methods_disp = ["Vanilla QR","QRW","SQR","Interval Score"]
methods_disp = dict(zip(methods,methods_disp))

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
for method in methods:
    for data_name in data_list:
        resultDir = "../../{}/results/{}/{}/".format(method,data_name,method)
        
        fileList = pd.DataFrame(os.listdir(resultDir),columns=['fileName'])
        fileList['seed'] = fileList['fileName'].apply(lambda x:int(x.split('_')[1]))
        fileList.sort_values(['seed'],inplace=True)
        
        fileName = fileList.groupby('seed',sort=False).first()['fileName'].tolist()[0]
        results = pd.read_csv(resultDir+fileName)
        quantiles = np.sort(results[results.columns[:-1]].values, axis=1)
        
        index = {'air':[147867],'naval':[16],'consumption':[4356]}
        xticks = {'air':[0.8,1.0,1.2,1.4,1.6,1.8],'naval':[-1.80,-1.75,-1.7,-1.65,-1.6],
                  'consumption':[-0.8,-0.6,-0.4,-0.2,0,0.2]}
        if data_name not in index:
            continue
        pdf = PdfPages("otherDensities/{}-{}.pdf".format(method,data_name))
        for ind in index[data_name]:
            plt.figure(figsize=(5.7,3.8),dpi=300)
            
            xs = quantiles[ind][1:-1]
            ys = 0.02/(quantiles[ind][2:]-quantiles[ind][:-2]+1e-8)
            plt.plot(xs, ys, linestyle='solid', color='b', linewidth=2)
            
            # plt.legend(loc='upper right', fontsize=20)
            plt.title("{} on {}".format(methods_disp[method],capitalize(data_name)), fontsize=20)
            plt.xticks(xticks[data_name], fontsize=20)
            plt.xlim((xticks[data_name][0],xticks[data_name][-1]))
            plt.yticks(fontsize=20)
            
            pdf.savefig()
            plt.close()
        pdf.close()

with open("otherDensities/tex.txt", 'w') as file:
    for method in methods:
        i = 1
        for data_name in ['naval','consumption','air']:
            file.write("\\begin{minipage}[t]{.33\\linewidth}\n")
            file.write("\\centerline{{\\includegraphics[width=\\linewidth]{{plots/otherDensities/{}-{}.pdf}}}}\n".format(method,data_name))
            if i == 3:
                file.write("\\end{minipage}\\\\\n")
            else:
                file.write("\\end{minipage}\\hfill\n")
            i = i+1



