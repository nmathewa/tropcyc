#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:26:04 2023

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
sst_data = xr.open_dataset(main_dir+'sst_new.nc').sst
mask_land = sst_data.notnull()

pres_data = xr.open_dataset(main_dir+'Spres_new.nc').sp.where(mask_land)

rh_data = xr.open_dataset(main_dir+'rh_mid_new.nc').r.where(mask_land)

vort_data = xr.open_dataset(main_dir+'vort_850_new.nc').vo.where(mask_land)

#%%


def plot_map():
    
    fig,ax = plt.subplots(figsize=(20,10),subplot_kw={"projection": ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)


    ax.coastlines()
    
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='black', alpha=0.1, linestyle='--')
    ax.set_extent([-100, 30, 0, 80], crs=ccrs.PlateCarree())
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    #data.plot(ax=ax,transform=ccrs.PlateCarree())

    return fig,ax

#%%


fig,ax = plot_map()

out_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Images/'

sst_data.mean(dim='time').plot(ax=ax)
ax.set_title('')

fig.savefig(out_dir+'ERA5_plot.png')

