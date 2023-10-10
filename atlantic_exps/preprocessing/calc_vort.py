#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:33:56 2023

@author: nmathewa
"""

import xarray as xr
import os 
import glob 
import pandas as pd
import numpy as np

#%%

def calc_mean_box(df,dset):
    lats,lons = df["LAT"],df["LON"]
    lonmin = lons - 2.5
    lonmax = lons + 2.5
    latmin = lats - 2.5
    latmax = lats + 2.5
    resolution = 0.25
    mean_vals = np.zeros(len(df))
    for ii in range(len(df)):
        lat = df["LAT"].iloc[ii]
        lon = df["LON"].iloc[ii]
        time = df["datetime"].iloc[ii]
        try:
            time_slice = dset.sel(time=time)
        except KeyError:
            continue
        
        lats = []
        lons = []
        for lat in np.arange(latmin.values[0], 
                             latmax.values[0], resolution):
            for lon in np.arange(lonmin.values[0], 
                                 lonmax.values[0], resolution):
                lats += [lat]
                lons += [lon]
    
        n_array = time_slice.sel(latitude=np.unique(lats),
                                 longitude=np.unique(lons),
                                 method='nearest')
        
        
    
        mean_val = n_array.mean().values
        
        mean_vals[ii] = mean_val
        
    
    
    
    return mean_vals
    
#%%

main_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'

vort_data = xr.open_mfdataset(main_dir+'vort850/*.nc').sel(expver=5).vo



#%%


ib_data = pd.read_csv(main_dir+'final_filtered_storms.csv',keep_default_na=False)

ib_data_tracks = ib_data[['SID','datetime','LAT','LON','USA_WIND']]

groups = ib_data_tracks.groupby("SID")

timed_frames = []

for cyc_id,cyclone in groups:
    
    cyclone['datetime'] = pd.to_datetime(cyclone['datetime'])
    
    timed = cyclone.set_index('datetime')
    
    n_timed = timed.resample('3H').first()
    
    timed_frames.append(n_timed)
    
ib_data_tracks2 = pd.concat(timed_frames).reset_index()

#%%

groups = ib_data_tracks2.groupby('SID')


dfts = []
for cyc_id , cyc_event in groups:
    print(cyc_id)
    n_dft = cyc_event
    
    n_dft['vort_850'] = calc_mean_box(n_dft,vort_data)
    
    dfts.append(n_dft)

final_dft = pd.concat(dfts)





