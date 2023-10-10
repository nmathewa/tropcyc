#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:52:53 2023

@author: nmathewa
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

os.chdir('/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/')

filter_dft = pd.read_csv('final_filtered_storms.csv',keep_default_na=False)



#%%

def plot_tracks(filtered_ib):
    filtered_ib['datetime'] = pd.to_datetime(filtered_ib['datetime'])

    events = filtered_ib.groupby('SID')
    
    fig,ax = plt.subplots(figsize=(20,10),subplot_kw={"projection": ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.COASTLINE)

    ax.add_feature(cfeature.BORDERS)


    ax.coastlines()
    
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    for event_num , event in events:
        roll_events = event.set_index('datetime')[['LON','LAT']].rolling(window=24).mean()
        lon = event['LON'].values
        lat = event['LAT'].values
        ax.plot(lon,lat,transform=ccrs.PlateCarree(),linewidth=1,color='black')
    return fig,ax

#%%

plot_tracks(filter_dft)