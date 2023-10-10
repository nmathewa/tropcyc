#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:22:14 2023

@author: nmathewa
"""

import xarray as xr
import os 
import glob 
import pandas as pd

#%%

ib_data = pd.read_csv('final_filtered_storms.csv',keep_default_na=False)

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



sst_data = xr.open_mfdataset('surfacePT/*.nc').sel(expver=5).sst

#%%

test_cyclone = ib_data_tracks[ib_data_tracks['SID'] == '2023264N13336']

times = test_cyclone['datetime']

sst_time_slice = sst_data.sel(time=times.values)

from shapely import Point

# thresh = 1000 km radius 
# 


lats = test_cyclone['LAT']
lons = test_cyclone['LON']

# 0.25 degrees ===> 1000 km is roughly 2.5

lonmin = lons - 2.5
lonmax = lons + 2.5
latmin = lats - 2.5
latmax = lats + 2.5
resolution = 0.25

lats = []
lons = []
for lat in np.arange(latmin.values[0], latmax.values[0], resolution):
    for lon in np.arange(lonmin.values[0], lonmax.values[0], resolution):
        lats += [lat]
        lons += [lon]

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


n_dft = ib_data_tracks2[ib_data_tracks2['SID'] == '2001227N13323']

n_dft['sst_mean'] = calc_mean_box(n_dft,sst_data)


#%%

groups = ib_data_tracks2.groupby('SID')


dfts = []
for cyc_id , cyc_event in groups:
    print(cyc_id)
    n_dft = cyc_event
    
    n_dft['temp_mean'] = calc_mean_box(n_dft,sst_data)
    
    dfts.append(n_dft)

final_temp_dft = pd.concat(dfts)

#%%


final_temp_dft.to_csv('sst_temp_test.csv')


