# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from evaluation import *

#==============================================================================
# Electrical Grid Stability

data = pd.read_csv('../data/Data_for_UCI_named.csv')
data = data.drop(['stabf'], axis = 1)
data = data.dropna()

var = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4']
y = np.array(['stab'])[0]

aver_calibrate("grid", data, var, y)

#==============================================================================
# Conditional Based Maintenance (naval)

data = pd.read_table('../data/CBM_data.txt', sep = '   ')
data.columns = ['lp', 'v', 'GTT', 'GTn', 'GGn','Ts','Tp','T48','T1','T2','P48','P1','P2','Pexh','TIC','mf','GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']
data = data.drop(['GT Turbine decay state coefficient', 'T1'], axis = 1)
data = data.dropna()

var = ['lp', 'v', 'GTT', 'GTn', 'GGn','Ts','Tp','T48','T2','P48','P1','P2','Pexh','TIC','mf']
y = np.array(['GT Compressor decay state coefficient'])[0]

aver_calibrate("naval", data, var, y)

#==============================================================================
# Appliances energy prediction

data = pd.read_csv('../data/energydata_complete.csv')
data = data.drop(['date'], axis = 1)
data = data.dropna()

var = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 
                     'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 
                     'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 
                     'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']
y = 'Appliances'
data[y] = np.log(data[y])

aver_calibrate("appliance", data, var, y)

#==============================================================================
# Real-time Election (election)

data = pd.read_csv('../data/ElectionData.csv')
data = data.drop(['TimeElapsed', 'time', 'territoryName', 'Party'], axis = 1)
data = data.dropna()

var = ['totalMandates', 'availableMandates', 'numParishes', 'numParishesApproved', 'blankVotes', 'blankVotesPercentage', 'nullVotes', 'nullVotesPercentage', 'votersPercentage', 'subscribedVoters', 'totalVoters', 'pre.blankVotes', 'pre.blankVotesPercentage', 'pre.nullVotes', 'pre.nullVotesPercentage', 'pre.votersPercentage', 'pre.subscribedVoters', 'pre.totalVoters', 'Mandates', 'Percentage', 'validVotesPercentage', 'Votes', 'Hondt']
y = np.array(['FinalMandates'])[0]

aver_calibrate("election", data, var, y)

#==============================================================================
# Industry Energy Consumption (steel)

data = pd.read_csv('../data/Steel_industry_data.csv')
data = data.drop(['date'], axis = 1)
data = data.dropna()

var = ['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM']
y =  np.array(['Usage_kWh'])[0]

aver_calibrate("steel", data, var, y)

#==============================================================================
# Facebook Comment Volume (facebook1)

data = pd.read_csv('../data/facebook1.csv')
data = data.dropna()

var = data.columns[0:52]
y = data.columns[-1]

aver_calibrate("facebook1", data, var, y)

#==============================================================================
# Facebook Comment Volume (facebook2)

data = pd.read_csv('../data/facebook2.csv')
data = data.dropna()

var = data.columns[0:52]
y = data.columns[-1]

aver_calibrate("facebook2", data, var, y)

#==============================================================================
# Beijing PM2.5 (PM2.5)

data = pd.read_csv('../data/PRSA_data_2010.1.1-2014.12.31.csv')
data = data.drop(['No'], axis = 1)
data = data.dropna()

var = ['year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is','Ir']
y = 'pm2.5'

aver_calibrate("pm2.5", data, var, y)

#==============================================================================
# Physicochemical Properties (bio)

data = pd.read_csv('../data/CASP.csv')
data = data.dropna()

var = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']
y = np.array(['RMSD'])[0]

aver_calibrate("bio", data, var, y)

#==============================================================================
# BlogFeedback (blog)

data = pd.read_csv('../data/blog.csv')
data = data.dropna()

var = data.columns[0:276]
y = data.columns[-1]

aver_calibrate("blog", data, var, y)

#==============================================================================
# Power consumption of T (consumption)

data = pd.read_csv('../data/Tetuan City power consumption.csv')
data = data.drop(['DateTime', 'Zone 2  Power Consumption', 'Zone 1 Power Consumption'], axis = 1)
data = data.dropna()

var = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
y = 'Zone 3  Power Consumption'

aver_calibrate("consumption", data, var, y)

#==============================================================================
# Online Video (video)

data = pd.read_csv('../data/transcoding_mesurment.tsv', sep = '\t')
data = data.dropna()

var = ['duration', 'width', 'height', 'bitrate', 'framerate', 'i', 'p', 'b',
       'frames', 'i_size', 'p_size', 'size', 'o_bitrate', 'o_framerate',
       'o_width', 'o_height']
y = np.array(['umem'])[0]

aver_calibrate("video", data, var, y)

#==============================================================================
# GPU kernel performance (gpu)

data = pd.read_csv('../data/sgemm_product.csv')
data = data.drop(['Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], axis = 1)
data = data.dropna()

var = ['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'MDIMA', 'NDIMB', 'KWI', 'VWM',
       'VWN', 'STRM', 'STRN', 'SA', 'SB']
y = np.array(['Run1 (ms)'])[0]

aver_calibrate("gpu", data, var, y)

#==============================================================================
# Query Analytics (query)

data = pd.read_csv('../data/Range-Queries-Aggregates.csv')
data = data.drop(['Unnamed: 0'], axis = 1)
data = data.dropna()

var = ['x', 'y', 'x_range', 'y_range', 'count', 'sum_']
y = np.array(['avg'])[0]

aver_calibrate("query", data, var, y)

#==============================================================================
# Wave Energy Converters (wave)

data = pd.read_csv('../data/wave.csv')
data = data.dropna()

var = data.columns[0:48]
y = data.columns[-1]

aver_calibrate("wave", data, var, y)

#==============================================================================
# Beijing Multi-Site Air-Quality Data (air)

data = pd.read_csv('../data/air.csv')
data = data.dropna()

var = ['year', 'month', 'day', 'hour', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
y = np.array(['PM2.5'])[0]

aver_calibrate("air", data, var, y)

#==============================================================================
# YearPredictionMSD (year)

data = pd.read_csv('../data/year.csv')
data = data.dropna()

var = data.columns[0:90]
y = np.array(['0'])[0]

aver_calibrate("year", data, var, y)
