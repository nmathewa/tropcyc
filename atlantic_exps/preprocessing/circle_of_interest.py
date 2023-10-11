#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 02:04:41 2023

@author: nmathewa
"""

import pandas as pd
import xarray as xr
import numpy as np


main_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'

ibdata = pd.read_csv(main_dir+'final_proc.csv')

test_data = ibdata[ibdata['SID'] == '2000175N09340']

lat = test_data['LAT'].iloc[0]
lon = test_data['LON'].iloc[0]

latmin = lat - 25
latmax = lat + 25
lonmin = lon - 25
lonmax = lon + 25

center = (lat,lon)

radius = 1000

#%%

lats = np.arange(latmin,latmax,0.25)
lons = np.arange(lonmin,lonmax,0.25)

lat_mesh,lon_mesh = np.meshgrid(lats,lons)

lats_l = np.radians(lats)
lons_l = np.radians(lons)

lat_meshR,lon_meshR = np.meshgrid(lats_l,lons_l)



distances = np.sqrt((center[0] - lat_mesh)**2 + (center[1] - lon_mesh)**2)

n_distance = distances * 111


latiy,lonix = np.where(n_distance <= 1000)


filtered_lats = lat_mesh.flatten()[latiy]
filtered_lons = lon_mesh.flatten()[latiy]






