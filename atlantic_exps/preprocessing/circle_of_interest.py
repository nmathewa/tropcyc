#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 02:04:41 2023

@author: nmathewa
"""

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
pd.options.mode.chained_assignment = None  # default='warn'
main_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'

ibdata = pd.read_csv(main_dir+'final_proc.csv')
ibdata['datetime'] = pd.to_datetime(ibdata['datetime'])

test_data = ibdata[ibdata['SID'] == '2001301N27325']




#%%

sst_data = xr.open_mfdataset(main_dir+'surfacePT/*.nc').sel(expver=5).sst
mask_land = sst_data.notnull()

pres_data = xr.open_mfdataset(main_dir+'surfacePT/*.nc').sel(expver=5).sp.where(mask_land)

rh_data = xr.open_mfdataset(main_dir+'RH_mid/*.nc').sel(expver=5).r.where(mask_land)

#%%




#%%
def angular_average(var_dset,lat,lon,vertical=False):
    latmin = int(lat) - 10
    latmax = int(lat) + 10
    lonmin = int(lon) - 10
    lonmax = int(lon) + 10
    center = (lat,lon)
    
    
    lats = np.arange(latmin,latmax,0.25)
    lons = np.arange(lonmin,lonmax,0.25)
    
    #print(len(lats))
    #print(len(lons))
    
    lat_mesh,lon_mesh = np.meshgrid(lats,lons)

    radius = 1000
    
    distances = np.sqrt((center[0] - lat_mesh)**2 + (center[1] - lon_mesh)**2)
    
    n_distance = distances * 111

    #print(distances.shape)
    
    dist_cond = n_distance <= radius
    
    cond_array = xr.DataArray(dist_cond,dims=['latitude','longitude'])

    if vertical == True:
        var_dset_sub = var_dset.sel(latitude=lats,longitude=lons,method='nearest').mean(dim='level')
    elif vertical == False:
        var_dset_sub = var_dset.sel(latitude=lats,longitude=lons,method='nearest')
        
        
    masked_var_data = var_dset_sub.where(cond_array)
    
    annul_mean = masked_var_data.mean()
    
    return annul_mean

#%%

lat,lon = test_data['LAT'].iloc[0],test_data['LON'].iloc[0]
testrh = angular_average(rh_data.isel(time=0),lat,lon,vertical=True)



#%%


groups = ibdata.groupby('SID')

#%%
dfts = []
ctr = 0
for event_id , event in groups:
    ctr = ctr + 1
    n_event = event.sort_values('lead')
    
    print(str(int((1/len(groups))*100)))
    all_means = []
    for ii in range(len(n_event)):
        
        time = n_event['datetime'].iloc[ii]
        
        lat = n_event['LAT'].iloc[ii]
        lon = n_event['LON'].iloc[ii]
        
        try :
            pres_data_t = pres_data.sel(time=time)
        except KeyError:
            amean_pres = np.nan
        
        
        amean_pres = angular_average(sst_data,lat,lon).values
        
        all_means += [amean_pres]
    
    n_event['sst_mean'] = np.array(all_means)
    
    dfts += [n_event]
    
final_arr = pd.concat(dfts)
        
        




#%%

    
fig,ax = plt.subplots(figsize=(20,10),subplot_kw={"projection": ccrs.PlateCarree()})
    
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)


ax.coastlines()
    
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
    
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_extent([-100, 30, 0, 80], crs=ccrs.PlateCarree())
testrh[1].plot(ax=ax,transform=ccrs.PlateCarree())


#%%


