#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:30:13 2023

@author: nmathewa
"""


import pandas as pd
import xarray as xr
import numpy as np

#main_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'

main_dir = '/Users/nalex2023/main/tropcyc/atlantic_exps/Datasets/'




ib_data = pd.read_csv(main_dir+'final_filtered_storms1980.csv').dropna(how='any',axis=1)








ib_data_tracks = ib_data[['SID','datetime','LAT','LON','USA_WIND','DIST2LAND']]



#rh_data = pd.read_csv(main_dir+'mean_rh.csv').dropna(how='any',axis=0)

#temp_data = pd.read_csv(main_dir+'sst_temp_test.csv')

#pres_data = pd.read_csv(main_dir+'test1_pres.csv')




#%%

groups = ib_data_tracks.groupby("SID")

timed_frames = []

for cyc_id,cyclone in groups:
    
    cyclone['datetime'] = pd.to_datetime(cyclone['datetime'])
    
    timed = cyclone.set_index('datetime')
    
    n_timed = timed.resample('3H').first()
    
    hour_ranges = np.arange(len(n_timed)) * 3
    
    n_timed['lead'] = hour_ranges
    
    n_timed['scale'] =pd.cut(n_timed['USA_WIND'],bins=[0,33,63,82,95,112,136,137],
                                 labels=['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'])
 
    
    chg_24 = n_timed[n_timed['lead'] <= 24]['USA_WIND']
    
    mean_change = chg_24.diff().mean()
    
    n_timed['pers'] = [mean_change]*len(n_timed) 
    
    n_timed_fil = n_timed.dropna(how='any',axis=0)
    
    timed_frames.append(n_timed_fil)
    
ib_data_tracks2 = pd.concat(timed_frames).reset_index()
#%%

ib_data_tracks2.DIST2LAND.plot()


#%%

test_event = ib_data_tracks2[ib_data_tracks2['SID'] == '2012255N16322'].reset_index()



saff_sim_Scale = pd.cut([0,33,63,82,95,112,136,137],
                        bins=7,labels=['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'])

test_event['scale'] = pd.cut(test_event['USA_WIND'],bins=[0,33,63,82,95,112,136,137],
                             labels=['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'])


hour_ranges = np.arange(len(test_event)) * 3

test_event['lead'] = hour_ranges


idx_24 = test_event[test_event['lead'] <= 24]

mean_diff = idx_24['USA_WIND'].diff().mean()



#%%


ib_data_tracks2.to_csv(main_dir+'final_proc1980.csv')





