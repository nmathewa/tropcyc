#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:27:38 2023

@author: nmathewa
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
import cartopy.feature as cfeature

from prepro_features import prepro_features


os.environ['CARTOPY_USER_BACKGROUNDS'] = '/home/nmathewa/main/GIT/doe_mjo_waves_paper/analysis1/modules/nasa_maps/'

ib_data = pd.read_csv('/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/datasets/proc_tracks.csv')

ib_data['datetime'] = pd.to_datetime(ib_data['datetime'])

ib_data_6h = ib_data.set_index('datetime').groupby('SID').resample('6H').first().droplevel(0).reset_index()

ib_data_6h = ib_data_6h.set_index('datetime').dropna(axis=0,how='all').reset_index()

ib_data_test = ib_data_6h[ib_data_6h['SID'] == '2022311N21293']

shear_data = xr.open_dataset(in_datasets+'vspeed_shear.nc').speed_shear



def plot_map(ibdata,dset):
    
    ibadata_sids = ibdata.groupby('SID')
    fig,ax = plt.subplots(figsize=(20,10),subplot_kw={"projection": ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)


    ax.coastlines()
    ax.background_img(name='BM', resolution='high')
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='black', alpha=0.1, linestyle='--')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_extent([-90, -30, 0, 40])

    lons = event['LON'].astype(float)
    lats = event['LAT'].astype(float)
    ax.plot(lons,lats,transform=ccrs.PlateCarree(),color='red',linewidth=2)
    
    for ii in range(len(ibdata)):
        time = ib_data.datetime.iloc[ii]
        lat = lats.iloc[ii]
        lon = lons.iloc[ii]
        n_t_dset = dset.sel(time=time)
        annul_box = aoi.angular_imgs(var_dset=n_t_dset,lat=lat,lon=lon,box=True)
        annul_box.plot(transform=ccrs.PlateCarree(),ax=ax,add_colorbar=False)
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    return fig,ax

fig,ax = plot_map(ib_data_test,shear_data)

test_out = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/datasets/images/test/'

fig.savefig(test_out+'track_box.png')

