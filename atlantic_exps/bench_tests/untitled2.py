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