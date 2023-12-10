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

#in_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/preprocessing/'

in_dir = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Preprocessing/'
stat_dft_unsort = pd.read_csv(in_dir+'final_events_v1_20y.csv',index_col=None)

stat_dft = stat_dft_unsort.sort_values(['cyclone_id','lead_time'])



#%%
imgs = []
cyc_ids = []
lead_times = []
for ii in range(len(stat_dft)):
    nc_file = stat_dft['files'].iloc[ii]
    dset = xr.open_dataset(nc_file)
    
    all_vars = list(dset.keys())
    
    all_arrs = []
    for jj in all_vars:
        dset_data = dset[jj].values
        all_arrs += [dset_data]
    
    try:
        n_data = np.dstack(all_arrs)
        
    except ValueError:
        print("file empty")
        continue
    if np.isnan(n_data).any():
        continue
    else:
        
        cyc_ids += [stat_dft['cyclone_id'].iloc[ii]]
        lead_times += [stat_dft['lead_time'].iloc[ii]]
        
        imgs += [np.dstack(all_arrs)]
        

    

#%%


final_images = np.stack(imgs,axis=0)


np.save(in_dir+'final_arrv3.npy',final_images)

#%%

support_file = pd.DataFrame(cyc_ids,columns=['id'])

support_file['lead_time'] = lead_times


#%% create targets 

support_file.to_csv(in_dir+'support_file3.csv')
#%%create targets 

in_events = '/Users/nalex2023/main/tropcyc/atlantic_exps2/datasets/proc_tracks.csv'

dft_speed = pd.read_csv(in_events)


target_order = []
for ii in range(len(support_file)):
    id_val = support_file['id'].iloc[ii]
    lead = support_file['lead_time'].iloc[ii]
    target_val = dft_speed[(dft_speed['SID'] == id_val) & (dft_speed['lead'] == lead)]
    target_order += [target_val['USA_WIND'].values[0]]


#%%

support_file['USA_WIND'] = target_order

support_file.to_csv(in_dir+'targetsv3.csv')

#%%


