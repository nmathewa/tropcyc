#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:36:36 2023

@author: nmathewa
"""
import pandas as pd
try:
    import cupy as np
    print('using cupy')
except ImportError:
    print('using numpy')
    import numpy as np

class ib_processor:
    '''
    inputs: csv location

    '''
    def __init__(self,csv_loc,all_cols=True):
        
        read_data = pd.read_csv(csv_loc,keep_default_na=False,low_memory=False).iloc[1:,:]
        read_data['datetime'] = pd.to_datetime(read_data['ISO_TIME'],format='%Y-%m-%d %H:%M:%S')

        req_cols = read_data[['datetime','SID','LAT','LON','USA_WIND','DIST2LAND','BASIN']]
        
        self.final_data = req_cols
        
    def filter_data(self,data=None,y1=1980,y2=2023,basin='NA'):
        print('Filtering data for the period {} to {}'.format(y1,y2))
        if data is None:
            fil_data = self.final_data
        else:
            fil_data = data
        
        year_mask = (fil_data['datetime'] > str(y1)) & (fil_data['datetime'] <= str(y2))
        period_new = fil_data[year_mask][fil_data['BASIN'] == basin]
        
        return period_new
    
    def filter_ET(self,df):
        ordered_df = df.sort_values(by='datetime')
        lat_filter = ordered_df['LAT'] <= 30
        filter_df = ordered_df[lat_filter]
        return filter_df
        
    def compute_cols(self,data):
        if data is None:
            fil_data = self.final_data
        else:
            fil_data = data
        timed_frames = []
        grouped_data = fil_data.groupby('SID')
        
        for cyc_id,cyclone in grouped_data:
            
            cyclone['datetime'] = pd.to_datetime(cyclone['datetime'])
            
            timed = cyclone.set_index('datetime')
            
            n_timed = timed.resample('3H').first()
            
            hour_ranges = np.arange(len(n_timed)) * 3
            
            n_timed['lead'] = hour_ranges
            
            #n_timed['scale'] =pd.cut(n_timed['USA_WIND'],bins=[0,33,63,82,95,112,136,137],
                                         #labels=['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5'])
         
            
            #chg_24 = n_timed[n_timed['lead'] <= 24]['USA_WIND']
            
            #mean_change = chg_24.diff().mean()
            
            #n_timed['pers'] = [mean_change]*len(n_timed) 
            
            n_timed_fil = n_timed.dropna(how='any',axis=0)
            
            timed_frames.append(n_timed_fil)
            
        ib_data_tracks2 = pd.concat(timed_frames).reset_index()
        
        filter_ets = ib_data_tracks2.groupby('SID').apply(self.filter_ET).reset_index(drop=True)
        
        return filter_ets
    
    
    
    