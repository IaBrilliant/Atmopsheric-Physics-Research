#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:10:34 2022

@author: brilliant
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
import os
import time as tm

''' Moving Average for Historical Data '''

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Observational Data:
years_obs, jet_latitude_obs, jet_strength_obs, NAO_index_obs = \
    np.loadtxt('/rds/general/user/.../home/.../parameters.txt', skiprows = 1, unpack = True)

# Detrending Data:     
NAO_index_obs_de = signal.detrend(NAO_index_obs)
Jet_strength_obs_de = signal.detrend(jet_strength_obs)
Jet_latitude_obs_de = signal.detrend(jet_latitude_obs)

# Calculating Moving Average (10 & 30 years); 
# mov_avg10 - 155-9 years of data
# mov_avg30 - 155-29 years of data
mov_avg10NAO_obs = np.array(moving_average(NAO_index_obs_de,10))
mov_avg30NAO_obs = np.array(moving_average(NAO_index_obs_de,30))
mov_avg10JS_obs = np.array(moving_average(Jet_strength_obs_de,10))
mov_avg30JS_obs = np.array(moving_average(Jet_strength_obs_de,30))
mov_avg10JL_obs = np.array(moving_average(Jet_latitude_obs_de,10))
mov_avg30JL_obs = np.array(moving_average(Jet_latitude_obs_de,30))

time10_obs = np.array([])
time30_obs = np.array([])

for i in range(0, len(mov_avg10NAO_obs)): 
    time10_obs = np.append(time10_obs, 1870+i)
for i in range(0, len(mov_avg30NAO_obs)): 
    time30_obs = np.append(time30_obs, 1890+i)
    
print(' Time Arrays are Created ')

#%%

''' Gradient Analysis for Historical Data '''

# NAO Gradient
# 30-year time-periods are studied
gradientNAO = np.array([])
for i in range(0, len(mov_avg10NAO_obs) - 30): 
    ma_30NAO = mov_avg10NAO_obs[i : (i+30)]
    dyNAO = (ma_30NAO[29] - ma_30NAO[0])
    gradientNAO = np.append(gradientNAO, dyNAO/30)

gradient_absNAO_obs = np.absolute(gradientNAO)
resultNAO = gradient_absNAO_obs[np.where(gradient_absNAO_obs == max(gradient_absNAO_obs))]
max_starting_yearNAO = np.where(gradient_absNAO_obs == resultNAO)
indexNAO_obs = max_starting_yearNAO[0][0]
yearNAO = int(years_obs[9]) + indexNAO_obs

# Jet Strength Gradient
gradientJS = np.array([])
for i in range(0, mov_avg10JS_obs.shape[0] - 30):
    ma_30JS = np.array([])
    ma_30JS = np.append(ma_30JS, mov_avg10JS_obs[i : (i+30)])
    dyJS = (ma_30JS[29] - ma_30JS[0])
    gradientJS = np.append(gradientJS, dyJS/30)

gradient_absJS_obs = np.absolute(gradientJS)
resultJS = gradient_absJS_obs[np.where(gradient_absJS_obs == max(gradient_absJS_obs))]
max_starting_yearJS = np.where(gradient_absJS_obs == resultJS)
indexJS_obs = max_starting_yearJS[0][0]
yearJS = int(years_obs[9]) + indexJS_obs

# Jet Latitude Gradient
gradientJL = np.array([])
for i in range(0,mov_avg10JL_obs.shape[0] - 30):
    ma_30JL = np.array([])
    ma_30JL = np.append(ma_30JL, mov_avg10JL_obs[i : (i+30)])
    dyJL = (ma_30JL[29] - ma_30JL[0])
    gradientJL = np.append(gradientJL, dyJL/30)

gradient_absJL_obs = np.absolute(gradientJL)
resultJL = gradient_absJL_obs[np.where(gradient_absJL_obs == max(gradient_absJL_obs))]
max_starting_yearJL = np.where(gradient_absJL_obs == resultJL)
indexJL_obs = max_starting_yearJL[0][0]
yearJL = int(years_obs[9]) + indexJL_obs

print(' Historical Data is Processed ')

''' Gradient Analysis for CMIP5 Data '''
# Same Process but for Every Model

path = "/rds/general/user/.../home/.../parameters_historical"
model_list = os.listdir(path)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (5,8), dpi = 100)
counter = 1

NAO_grad_hist = np.array([])
JL_grad_hist = np.array([])
JS_grad_hist = np.array([])

for model in model_list:
    if model[0] == '.': #Correction to an Error
        continue
    
    print('%s initiated: %d/%d' % (model[11:20], counter, len(model_list)))
    
    years, jet_latitude, jet_strength, NAO_index = np.loadtxt('%s/%s' % (path, model), skiprows = 1, unpack = True)
    NAO_index_de = signal.detrend(NAO_index)
    Jet_strength_de = signal.detrend(jet_strength)
    Jet_latitude_de = signal.detrend(jet_latitude)
    mov_avg10NAO = np.array(moving_average(NAO_index_de, 10))
    mov_avg10JS = np.array(moving_average(Jet_strength_de, 10))
    mov_avg10JL = np.array(moving_average(Jet_latitude_de, 10))

    gradientNAO = np.array([])
    for i in range(0, len(mov_avg10NAO)-30):
        ma_30NAO = np.array([])
        ma_30NAO = np.append(ma_30NAO, mov_avg10NAO[i:(i+30)])
        dyNAO = (ma_30NAO[29]-ma_30NAO[0])
        gradientNAO = np.append(gradientNAO, dyNAO/30)
    gradient_absNAO = np.absolute(gradientNAO)
    
    resultNAO = gradient_absNAO[np.where(gradient_absNAO == max(gradient_absNAO))]
    NAO_grad_hist = np.append(NAO_grad_hist, max(gradient_absNAO))
    max_starting_yearNAO = np.where(gradient_absNAO == resultNAO)
    indexNAO = max_starting_yearNAO[0][0]
    yearNAO = int(years[9]) + indexNAO

    max_linear_trendNAO = np.array([])
    for i in range(indexNAO, indexNAO+30): 
        max_linear_trendNAO = np.append(max_linear_trendNAO, mov_avg10NAO_obs[i])
    ax1.plot(years[9:], mov_avg10NAO, color = 'grey' )

    gradientJS = np.array([])
    for i in range(0,mov_avg10JS.shape[0]-30):
        ma_30JS = np.array([])
        ma_30JS = np.append(ma_30JS, mov_avg10JS[i:(i+30)])
        dyJS = (ma_30JS[29]-ma_30JS[0])
        gradientJS = np.append(gradientJS, dyJS/30)
    gradient_absJS = np.absolute(gradientJS)
    
    resultJS = gradient_absJS[np.where(gradient_absJS == max(gradient_absJS))]
    JS_grad_hist = np.append(JS_grad_hist, max(gradient_absJS))
    max_starting_yearJS = np.where(gradient_absJS == resultJS)
    indexJS = max_starting_yearJS[0][0]
    yearJS = int(years[9])+indexJS

    max_linear_trendJS = np.array([])
    for i in range(indexJS, indexJS+30): 
        max_linear_trendJS = np.append(max_linear_trendJS, mov_avg10JS[i])

    gradientJL = np.array([])
    for i in range(0,mov_avg10JL.shape[0]-30):
        ma_30JL = np.array([])
        ma_30JL = np.append(ma_30JL, mov_avg10JL[i:(i+30)])
        dyJL = (ma_30JL[29]-ma_30JL[0])
        gradientJL = np.append(gradientJL, dyJL/30)
    gradient_absJL = np.absolute(gradientJL)
    
    resultJL = gradient_absJL[np.where(gradient_absJL == max(gradient_absJL))]
    JL_grad_hist = np.append(JL_grad_hist, max(gradient_absJL))
    max_starting_yearJL = np.where(gradient_absJL == resultJL)
    indexJL = max_starting_yearJL[0][0]
    print(indexJL)
    yearJL = int(years[9])+indexJL

    max_linear_trendJL = np.array([])
    for i in range(indexJL, indexJL+30): 
        max_linear_trendJL = np.append(max_linear_trendJL, mov_avg10JL[i])
    ax2.plot(years[9:], mov_avg10JL, color = 'grey' ) 
    ax3.plot(years[9:], mov_avg10JS, color = 'grey')
    print('%s done %d/%d' % (model[11:20], counter, len(model_list) ))
    counter+=1
 
#%%

''' Plotting Results ''' 
   
ax1.plot(time10_obs, mov_avg10NAO_obs, color = 'black', linestyle = 'dashed')
ax1.plot(time30_obs, mov_avg30NAO_obs, color = 'black', linestyle = 'dotted') 
#ax1.plot(years_obs[(9+int(indexNAO)):(39+int(indexNAO)], max_linear_trendNAO, color ='black')
ax1.plot(time10_obs[(indexNAO_obs):(30+indexNAO_obs)], mov_avg10NAO_obs[indexNAO_obs:(indexNAO_obs+30)], color ='black')
#ax1.plot(time10_CMIP5_1, mov_avg10NAO_CMIP5_1, color = 'gray' )
ax1.set_ylim([-20, 20])
ax1.set_xlim([1860, 2015])
ax1.set_ylabel('NAO Index')
ax1.set_title('Smoothed NAO Index over time')
ax2.plot(time10_obs, mov_avg10JL_obs, color = 'black', linestyle = 'dashed')
ax2.plot(time30_obs, mov_avg30JL_obs, color = 'black', linestyle = 'dotted')
ax2.plot(time10_obs[(indexJL_obs):(30+indexJL_obs)], mov_avg10JL_obs[indexJL_obs:(indexJL_obs+30)], color ='black')
#ax2.plot(time_maxJL, max_linear_trendJL, color ='black')
#ax2.plot(time10_CMIP5_1, mov_avg10JL_CMIP5_1, color = 'gray' )
ax2.set_ylim([-10, 10])
ax2.set_xlim([1860, 2015])
ax2.set_ylabel('Smoothed Jet Latitude')
ax2.set_title('Smoothed Jet Latitude over time')
ax3.plot(time10_obs, mov_avg10JS_obs, color = 'black', linestyle = 'dashed')
ax3.plot(time30_obs, mov_avg30JS_obs, color = 'black', linestyle = 'dotted')
ax3.plot(time10_obs[(indexJS_obs):(30+indexJS_obs)], mov_avg10JS_obs[indexJS_obs:(indexJS_obs+30)], color ='black')
#ax3.plot(time_maxJS, max_linear_trendJS, color ='black')
#ax3.plot(time10_CMIP5_1, mov_avg10JS_CMIP5_1, color = 'gray' )
ax3.set_ylim([-3,3])
ax3.set_xlim([1860, 2015])
ax3.set_ylabel('Smoothed Jet Strength')
ax3.set_title('Smoothed Jet Strength over time')
ax3.set_xlabel('Time, years')
plt.tight_layout()
plt.savefig('Figure2_1*.png')
plt.show


fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (5,8), dpi = 100)
ax1.hist(NAO_grad_hist, density = True, bins = 11, color = 'grey', stacked = 1)
ax1.vlines(max(gradient_absNAO_obs), 0, 10)
ax1.set_xlim([0, 0.6])
ax1.set_ylim([0, 9])
#ax1.hist(max_linear_trendNAO_CMIP5_1)
ax1.set_title('30-yr Linear Trend in DJF NAO Index (hPA/yr)')
ax2.hist(JL_grad_hist, bins = 10, density = True, color = 'grey', stacked = 1)
ax2.vlines(max(gradient_absJL_obs), 0, 18)
ax2.set_xlim([0, 0.3])
ax2.set_ylim([0, 16])
#ax2.hist(max_linear_trendJL_CMIP5_1, color = 'black')
ax2.set_title('30-yr Linear Trend in DJF jet latitude (deg/yr)')
ax3.hist(JS_grad_hist, bins = 9, density = True, color = 'grey', stacked = 1)
ax3.vlines(max(gradient_absJS_obs), 0, 54)
ax3.set_xlim([0, 0.1])
ax3.set_ylim([0, 52])
#ax3.hist(max_linear_trendJS_CMIP5_1, color = 'black')
ax3.set_title('30-yr Linear Trend in DJF jet strength (m/s/yr)')
plt.tight_layout()
plt.savefig('Figure2_2*.png')
plt.show()