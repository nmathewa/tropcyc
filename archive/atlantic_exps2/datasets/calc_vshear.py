#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:47:06 2023
calc vertical shear
@author: nmathewa
"""

import xarray as xr
#loc = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/u_comp/'
#vloc = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/v_comp/'



#%%



#%%

def preprocess(uvdata):
    try :
        data = uvdata.sel(expver=5)
    except KeyError:
        data = uvdata
    
    meanup = data.sel(level=[200,250,300]).mean(dim='level')
    meanlow = data.sel(level=[775,800,925]).mean(dim='level')
    
    shear_c = abs(meanup - meanlow)
    
    return shear_c

#dset_u = xr.open_mfdataset(loc+'*.nc',preprocess=preprocess,
                           #parallel=True,chunks={'time':200})
dset_v = xr.open_mfdataset(vloc+'*.nc',preprocess=preprocess,
                           parallel=True,chunks={'time':200})

dset_v.to_netcdf('/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/v_shear.nc')

#%%
import numpy as np
import xarray as xr

v_data = xr.open_dataset('/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/v_shear.nc',chunks={'time':500})
u_data = xr.open_dataset('/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/u_shear.nc',chunks={'time':500})


speed_shear = np.sqrt(u_data*u_data + v_data*v_data)




