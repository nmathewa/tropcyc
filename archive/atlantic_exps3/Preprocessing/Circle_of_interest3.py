#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:49:32 2023

@author: nalex2023
"""
import xarray as xr 

dset1 = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Datasets/ERA5_processed_40/adaptor.mars.internal-1702325207.7002556-19708-6-41483b33-114e-4397-b26d-e9ee16020459.nc'

dset2 = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Datasets/ERA5_processed_40/adaptor.mars.internal-1702329804.8039672-8196-11-d6b8feef-5973-4b9d-a5ea-4161c42068a7.nc'

in_datasets = '/Users/nalex2023/main/tropcyc/atlantic_exps/Datasets/'


u = xr.open_dataset(dset1)

v = xr.open_dataset(dset2)
sst_20_23 = in_datasets+'sst_new.nc'

dset_test = xr.open_dataset(sst_20_23)

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
dset_v = xr.open_mfdataset(dset2,preprocess=preprocess,
                           parallel=True,chunks={'time':200})


#dset_v.to_netcdf('/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/v_shear.nc')



dsetv_80_99 = dset_v.sel(time=slice("1980","2000")).reindex(latitude=dset_test.latitude,
                                              longitude=dset_test.longitude).v


dset_u = xr.open_mfdataset(dset1,preprocess=preprocess,
                           parallel=True,chunks={'time':200})

dsetu_80_99 = dset_u.sel(time=slice("1980","2000")).reindex(latitude=dset_test.latitude,
                                              longitude=dset_test.longitude).u

shear_data = xr.open_dataset(in_datasets+'vspeed_shear.nc').speed_shear.reindex(time=dset_test.time,
                                                                                latitude=dset_test.latitude,
                                                                                longitude=dset_test.longitude)
out_fol = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Datasets/ERA5_processed_40/'


dsetv_80_99.to_netcdf(out_fol+'v_data80_99.nc')
#%%


dsetu_80_99.to_netcdf(out_fol+'u_data80_99.nc')

#%%

v_sh = xr.open_dataset(out_fol+'v_data80_99.nc')

u_sh = xr.open_dataset(out_fol+'u_data80_99.nc')

import numpy as np

shear = np.sqrt((v_sh.v*v_sh.v) + (u_sh.u*u_sh.u))

#%%

n_data = xr.merge([shear.rename('speed_shear'),shear_data], compat="no_conflicts")

#%%


#%%

#%%
out_fol = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Datasets/ERA5_processed_40/'


n_data.to_netcdf(out_fol+'shear_data40y.nc')