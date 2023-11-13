#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:18:46 2023

@author: nmathewa
"""

test_ins = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/datasets/images/test/'

import pandas as pd
import xarray as xr 

import glob 
import os
import matplotlib.pyplot as plt
all_ncs = glob.glob(test_ins+"*.nc")





#%%
for ii in all_ncs:
    names = ii.split(os.sep)[-1].split('.')[0] + '.png'
    #dset = xr.open_dataset(ii)
    #fig,ax = plt.subplots()
    #dset.sst.plot(ax=ax)
    #fig.savefig(test_ins+names)
#%%

fig,ax = plt.subplots(2,3)
axs = ax.flatten()
all_keys = list(dset.keys())
for ii in range(len(all_keys)):
    dset[all_keys[ii]].plot(ax=axs[ii])
    axs[ii].set_title(all_keys[ii])
