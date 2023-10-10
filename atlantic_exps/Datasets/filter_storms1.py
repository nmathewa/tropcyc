#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:07:58 2023

@author: nmathewa
"""

import pandas as pd
import os 
from global_land_mask import globe
import numpy as np

os.chdir("/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/")

dft = pd.read_csv('ibtracs.NA.list.v04r00.csv',keep_default_na=False)

units = dft.iloc[0,:]

n_dft = dft.iloc[1:,:]
#%%

n_dft['datetime'] = pd.to_datetime(n_dft['ISO_TIME'],format='%Y-%m-%d %H:%M:%S')

year_mask = (n_dft['datetime'] > '2000-1-1') & (n_dft['datetime'] <= '2023-12-31')
period_new = n_dft[year_mask][n_dft['BASIN'] == 'NA']


def only_na_basin(df):
    lon_wise = df.sort_values(by='datetime')
    
    if lon_wise['LON'].iloc[0] > -55:
        return df
    
    else:
        return None
    

only_neatlantic = period_new.groupby('SID').apply(only_na_basin).reset_index(drop=True)

counts = only_neatlantic.groupby('SID').count()


#%% storms only persist more than 24 hours


counts = only_neatlantic.groupby('SID').count().iloc[:,0]

counts_12 = counts[counts > 12].index

persist_storms = period_new[period_new['SID'].isin(counts_12)]


persist_storms['month']= persist_storms['datetime'].dt.month
#%% 

tesst = persist_storms[persist_storms['NUMBER'] == 40]


def mask_lands(df):
    ordered_df = df.sort_values(by='datetime')
    lat = ordered_df['LAT']
    lon = ordered_df['LON']
    ocean_mask = pd.Series(globe.is_ocean(lat=lat,lon=lon))
    idx_false = ocean_mask.idxmin()
    if idx_false == 0:
        return df
    else:
        land_mask = ocean_mask.iloc[:idx_false]
        final_masked = ordered_df.iloc[:idx_false,:]
        return final_masked



# if it creates a landfall for the first time drop over there


def filter_ET(df):
    ordered_df = df.sort_values(by='datetime')
    lat_filter = ordered_df['LAT'] <= 30
    filter_df = ordered_df[lat_filter]
    return filter_df

exclude_et = persist_storms.groupby('SID').apply(filter_ET).reset_index(drop=True)

ocean_only = exclude_et.groupby('SID').apply(mask_lands).reset_index(drop=True)




#%%

req_cols = ['SID','SEASON','NUMBER','BASIN','SUBBASIN',
            'NAME','datetime','NATURE','LAT','LON',
            'DIST2LAND','LANDFALL','STORM_SPEED','STORM_DIR','USA_WIND']

final_storms = ocean_only[req_cols]


final_storms.to_csv('final_filtered_storms.csv')






