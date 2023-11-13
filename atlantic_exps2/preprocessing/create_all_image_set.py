#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:25:17 2023

@author: nmathewa
"""


import xarray as xr
import pandas as pd
import numpy as np
import glob
import os

in_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/preprocessing/'

stat_dft = pd.read_csv(in_dir+'final_events.csv',index_col=None)

imgs = []
cyc_ids = []
lead_times = []
for ii in range(len(stat_dft)):
    nc_file = stat_dft['files'].iloc[ii]
    dset = xr.open_dataset(nc_file)
    all_vars = list(dset.keys())
    all_arrs = []
    for jj in all_vars:
        all_arrs += [dset[jj].values]
    try:
        imgs += [np.dstack(all_arrs)]
        cyc_ids += [stat_dft['cyclone_id'].iloc[ii]]
        lead_times += [stat_dft['lead_time'].iloc[ii]]
    except ValueError:
        print("file empty")
        continue
        
    

#%%

import tensorflow as tf


final_images = np.stack(imgs,axis=0)


final_image.save()





