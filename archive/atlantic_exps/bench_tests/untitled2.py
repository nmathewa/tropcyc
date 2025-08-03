#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:22:44 2023

@author: nmathewa
"""

import pandas as pd
import seaborn as sbs
import matplotlib.pyplot as plt

main_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'

out_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Images/'

dataset_ori = pd.read_csv(main_dir+'final3333.csv')


#%%

sbs.set_style('darkgrid')
#%%
dataset_ori['pres_mean'] = dataset_ori['pres_mean']/100

fig,ax = plt.subplots(2,3,figsize=(20,14))
axs = ax.flatten()

sbs.regplot(data=dataset_ori,x='USA_WIND',y='pres_mean',ax=axs[0],
            line_kws = {'color':'black'})


sbs.regplot(data=dataset_ori,x='USA_WIND',y='sst_mean',ax=axs[1],
            line_kws = {'color':'black'})

sbs.regplot(data=dataset_ori,x='USA_WIND',y='rh_data',ax=axs[2],
            line_kws = {'color':'black'})

sbs.regplot(data=dataset_ori,x='USA_WIND',y='vo_data',ax=axs[3],
            line_kws = {'color':'black'})


sbs.regplot(data=dataset_ori,x='USA_WIND',y='cor_param',ax=axs[4],
            line_kws = {'color':'black'})


sbs.regplot(data=dataset_ori,x='USA_WIND',y='pers',ax=axs[5],
            line_kws = {'color':'black'})

axs[0].set_title('Surface Pressure')
axs[1].set_title('Sea surface temperature')


axs[2].set_title('Mid Tropospheric humidity')
axs[3].set_title('Vorticity')
axs[4].set_title('Coriolis parameter')
axs[5].set_title('Persistence')


fig.savefig(out_dir+'correlationsofpredictors.png')


#%%multicolinearity 
import numpy as np

req_cols = ['pres_mean','sst_mean','rh_data',
                        'vo_data','cor_param',
                        'pers']
predicts = dataset_ori[req_cols]

corrs = predicts.corr()

mask = np.zeros_like(corrs, dtype=bool)



# Initialize matplotlib figure
fig, ax = plt.subplots(figsize=(4, 3))

# Generate a custom diverging colormap
cmap = sbs.diverging_palette(220, 10, as_cmap=True, sep=100)
cmap.set_bad('grey')

# Draw the heatmap with the mask and correct aspect ratio
sbs.heatmap(corrs, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)
fig.suptitle('Pearson correlation coefficient matrix', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=10)

#%%

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 

req_cols = ['SID','lead','pres_mean','sst_mean','rh_data',
                        'vo_data','cor_param',
                        'pers','USA_WIND']
n_data = dataset_ori[req_cols].dropna(how='any',axis=0)

unique_sids = n_data['SID'].unique()

train_sids, test_sids = train_test_split(unique_sids,test_size=0.2,random_state=1)

train_data = n_data[n_data['SID'].isin(train_sids)].drop('SID',axis=1)

test_data = n_data[n_data['SID'].isin(test_sids)].drop('SID',axis=1)



X_train = train_data.drop('USA_WIND',axis=1).values

y_train = train_data['USA_WIND'].values


X_test = test_data.drop('USA_WIND',axis=1).values
y_test = test_data['USA_WIND'].values

#%%

model = LinearRegression() 

model.fit(X_train, y_train) 

predictions = pd.DataFrame(model.predict(X_test),columns=['predicted'])

predictions['observed'] = y_test
#mae_error = abs(y_test - predictions)

#%%
import scipy as sp


fig,ax = plt.subplots(figsize=(20,12))
sbs.regplot(data=predictions,x='predicted',y='observed',ax=ax)
r, p = sp.stats.pearsonr(predictions['predicted'], predictions['observed'])

ax.text(.1, .8, 'r={:.2f}'.format(r),
            transform=ax.transAxes,fontsize=12)

fig.savefig(out_dir + 'Linear_model_bench.png')


