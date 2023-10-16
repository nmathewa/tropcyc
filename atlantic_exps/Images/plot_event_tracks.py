#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 09:51:30 2023

@author: nmathewa
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
import cartopy.feature as cfeature

data_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'
out_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Images/'


def plot_map(ibdata):
    
    ibadata_sids = ibdata.groupby('SID')
    fig,ax = plt.subplots(figsize=(20,10),subplot_kw={"projection": ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)


    ax.coastlines()
    
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='black', alpha=0.1, linestyle='--')
    ax.set_extent([-100, 0, 0, 40], crs=ccrs.PlateCarree())
    for sid,event in ibadata_sids:
        lons = event['LON'].astype(float)
        lats = event['LAT'].astype(float)
        ax.plot(lons,lats,transform=ccrs.PlateCarree(),color='black',linewidth=0.5)
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    return fig,ax

#%%

dft = pd.read_csv(data_dir+'ibtracs.NA.list.v04r00.csv',keep_default_na=False).iloc[1:,:]
dft['datetime'] = pd.to_datetime(dft['ISO_TIME'],format='%Y-%m-%d %H:%M:%S')
year_mask = (dft['datetime'] > '2000-1-1') & (dft['datetime'] <= '2023-12-31')
period_new = dft[year_mask][dft['BASIN'] == 'NA']

#%%
fig,ax = plot_map(period_new)



fig.savefig(out_dir+'all_nh_events.png',bbox_inches='tight',dpi=300,transparent=True)

#%% filtered events 

dft = pd.read_csv(data_dir+'final3333.csv')

fig,ax = plot_map(dft)

#ax.set_title('Fileterd North Atlantic Hurricane Events from 2020-2023',fontsize=20)

fig.savefig(out_dir+'filtered_nh_events.png',bbox_inches='tight',dpi=300,transparent=True)

