#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:07:58 2023

@author: nmathewa
"""

import pandas as pd
import os 


os.chdir("/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/")

dft = pd.read_csv('ibtracs.NA.list.v04r00.csv',keep_default_na=False)

units = dft.iloc[0,:]

n_dft = dft.iloc[1:,:]
#%%

n_dft['datetime'] = pd.to_datetime(n_dft['ISO_TIME'],format='%Y-%m-%d %H:%M:%S')

year_mask = (n_dft['datetime'] > '2000-1-1') & (n_dft['datetime'] <= '2023-12-31')
period_new = n_dft[year_mask][n_dft['BASIN'] == 'NA']

#%% storms only persist more than 24 hours


counts = period_new.groupby('NUMBER').count().iloc[:,0]

counts_12 = counts[counts > 12].index

persist_storms = period_new[period_new['NUMBER'].isin(counts_12)]


persist_storms['month']= persist_storms['datetime'].dt.month
#%% 

req_cols = ['SEASON','NUMBER','BASIN','SUBBASIN',
            'NAME','datetime','NATURE','LAT','LON',
            'DIST2LAND','LANDFALL','STORM_SPEED','STORM_DIR']

final_storms = persist_storms[req_cols]


final_storms.to_csv('final_filtered_storms.csv')






