#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:13:23 2023

@author: nalex2023
"""

import xarray as xr
in_datasets = '/Users/nalex2023/main/tropcyc/atlantic_exps/Datasets/'


sst_20_23 = in_datasets+'sst_new.nc'


rh_20_23 = in_datasets+'rh_mid_new.nc'

vort_20_23 = in_datasets+'vort_850_new.nc'

pres_20_23 = in_datasets+'Spres_new.nc'

dset_raw_old_sst_pres = '/Users/nalex2023/Temp/MTH_DEEP/sst_pres_era5_40_99.nc'

dset_sst_old = xr.open_dataset(dset_raw_old_sst_pres).sel(time=slice('1980','2000'))
dset_test = xr.open_dataset(sst_20_23)

sst_40_20 = dset_sst_old.sst.reindex(latitude=dset_test.latitude,
                                              longitude=dset_test.longitude)


#%%

out_fol = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Datasets/ERA5_processed_40/'

dset_all_sst = xr.concat([sst_40_20,dset_test.sst],dim='time')


dset_all_sst.to_netcdf(out_fol+'sst_data_40y.nc')
#%%

pres_80 = dset_sst_old.sp.reindex(latitude=dset_test.latitude,
                                              longitude=dset_test.longitude)

#%%


spres_23 = xr.open_dataset(pres_20_23) 

#%%

pres_all = xr.concat([pres_80,spres_23.sp],dim='time')


pres_all.to_netcdf(out_fol + 'spres_data_40y.nc')
#%%


rh_new = xr.open_dataset(rh_20_23)



rh_old_raw = '/Users/nalex2023/Temp/MTH_DEEP/era5_40_99_RH500.nc'
rh_old = xr.open_dataset(rh_old_raw).sel(time=slice('1980','2000')).r

rh_old_re = rh_old.reindex(latitude=dset_test.latitude,
                                              longitude=dset_test.longitude)



#%%

new_rh = xr.concat([rh_old_re,rh_new.r],dim='time')

#rh_old = xr.open_dataset()

#rh_all = xr.concat([rh_])


#cor_data =  2 * 7.29 * 1e-5 * np.sin(pres_20_23_data['latitude'])
#%%

new_rh.to_netcdf(out_fol+'RH_data40y.nc')

#%%



