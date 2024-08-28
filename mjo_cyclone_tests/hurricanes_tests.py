#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:16:01 2024

@author: nalex2023
"""

import pandas as pd
import datetime

#%%

data = pd.read_csv('/home/nalex2023@fit.edu/Downloads/ibtracs.ALL.list.v04r01.csv',
                   skiprows=0,keep_default_na=None)[1:]

data['ISO_TIME'] = pd.to_datetime(data['ISO_TIME'])

data['LAT'] = data['LAT'].astype(float)
data['LON'] = data['LON'].astype(float)

#%%


WP_basin = (data[data['BASIN'] == 'NI']).set_index('ISO_TIME')



grp_sid = WP_basin.groupby('SID').first().reset_index()
#grp_sid_NH =grp_sid['LAT'] < 0 

grp_sid_NH = grp_sid[grp_sid['LAT'] > 0]


#%%plotting



NI_basin = (data[data['BASIN'] == 'NI']).set_index('ISO_TIME')

SI_basin = (data[data['BASIN'] == 'SI']).set_index('ISO_TIME')


combined_frame = NI_basin.join(SI_basin,lsuffix='l')


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


subset_data = WP_basin



def plot_tracks(filtered_ib):
    #filtered_ib['datetime'] = pd.to_datetime(filtered_ib['ISO_TIME'])

    events = filtered_ib.groupby('SID')
    
    fig,ax = plt.subplots(figsize=(20,10),subplot_kw={"projection": ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.COASTLINE)

    ax.add_feature(cfeature.BORDERS)


    ax.coastlines()
    
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    for event_num , event in events:
        roll_events = event[['LON','LAT']]#.rolling(freq='1D').mean()
        #roll_eventsl = event[['LONl','LATl']].rolling(window=24).mean()
        lon = event['LON'].values
        lat = event['LAT'].values
        #lonl = event['LONl'].values
        #latl = event['LATl'].values
        ax.plot(lon,lat,transform=ccrs.PlateCarree(),linewidth=3,color='black')
        #ax.plot(lonl,latl,transform=ccrs.PlateCarree(),linewidth=3,color='black')
        
    return fig,ax

plot_tracks(SI_basin.loc['2002'])


#%%





#%%




diff = abs(grp_sid['ISO_TIME'].iloc[0] - grp_sid['ISO_TIME'].iloc[1]) < datetime.timedelta(days=9)

