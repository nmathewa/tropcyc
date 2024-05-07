#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:46:10 2023

@author: nalex2023
"""

import pandas as pd


class preprocess_cyclones:
    
    
    def __init__(self,ibtracks_dft):
        # reading ibtracks dataframe
        ib_data = pd.read_datagrame(ibtracks_dft,keep_default_na=False)
        
        ib_data['datetime'] = pd.to_datetime(ib_data['ISO_TIME'],format='%Y-%m-%d %H:%M:%S')
        
        self.ib_data = ib_data
    
    def filter_dates(self,start,end):
        # start,end as format '2023-12-31'
        
        if type(start) or type(end) != str:
            raise Exception(" start,end should be in format '2023-12-31'")
        dft = self.ib_data
        t_ib_data = (dft['datetime'] > start) & (dft['datetime'] <= end)
    
        return t_ib_data
    
    
    def filter_region(self,lat_min):
        gp_data = self.grouped_idx
        if gp_data['LAT'].iloc[0] > lat_min:
            return gp_data
        
    def filter_basin(self,basin):
        n_dft = self.ib_data
        
        b_n_dft = n_dft[n_dft['BASIN'] == basin]
        
        return b_n_dft
    
    def lead_time_filter(lead,freq=3):
        # lead time minimum required in hours 
        # freq of the data default is 3 hour
        steps = int(lead/freq)
        
        gp_data = self.grouped_idx
        
        if len(gp_data) < steps :
            return None
        else :
            return gp_data
    
    
     
            

    