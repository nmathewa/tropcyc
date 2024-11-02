#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 09:40:26 2023

@author: nmathewa
"""

import xarray as xr
import numpy as np


class prepro_features:
    
    def __init__(self):
        
        pass
    
    def angular_imgs(self,var_dset,lat,lon,radius=1000,test=False,box=False):
        if isinstance(var_dset,float):
            print('float based boundaries not supported')
            raise NotImplementedError
        
        else :
            latmin = int(lat) - 5
            latmax = int(lat) + 5
            lonmin = int(lon) - 5
            lonmax = int(lon) + 5
            center = (lat,lon)
        
        
            lats = np.arange(latmin,latmax,0.25)
            lons = np.arange(lonmin,lonmax,0.25)
        
            #print(len(lats))
            #print(len(lons))
        
            lat_mesh,lon_mesh = np.meshgrid(lats,lons)

            #radius = 1000
            radius = radius
        
            distances = np.sqrt((center[0] - lat_mesh)**2 + (center[1] - lon_mesh)**2)
        
            n_distance = distances * 111

            #print(distances.shape)
        
            dist_cond = n_distance <= radius
        
            cond_array = xr.DataArray(dist_cond,dims=['latitude','longitude'])


            var_dset_sub = var_dset.sel(latitude=lats,longitude=lons,method='nearest')
            
            if box :
                
                masked_var_data = var_dset_sub
            else :
                masked_var_data = var_dset_sub.where(cond_array)
            
            
            return masked_var_data