#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 09:49:02 2023

@author: nmathewa
"""

import os
os.chdir('/home/nmathewa/main/GIT/tropcyc/modules/')
import rioxarray as rio
from process_ib import ib_processor
from prepro_features import prepro_features
import xarray as xr
import pandas as pd
in_datasets = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'

sh_data = xr.open_dataset(in_datasets+'rh_mid_new.nc').r#.isel(time=1)

sst_data = xr.open_dataset(in_datasets+'sst_new.nc').sst.reindex(time=sh_data.time)

shear_data = xr.open_dataset(in_datasets+'vspeed_shear.nc').speed_shear.reindex(time=sh_data.time,
                                                                                latitude=sh_data.latitude,
                                                                                longitude=sh_data.longitude)

vort_data = xr.open_dataset(in_datasets+'vort_850_new.nc').vo.sel(expver=5).reindex(time=sh_data.time)

pres_data = xr.open_dataset(in_datasets+'Spres_new.nc').sp.reindex(time=sh_data.time)

#%%

#temp_data = xr.open_dataset(in_datasets+'rh_mid_new.nc')


ib_data = pd.read_csv('/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/datasets/proc_tracks.csv')

ib_data['datetime'] = pd.to_datetime(ib_data['datetime'])

ib_data_6h = ib_data.set_index('datetime').groupby('SID').resample('6H').first().droplevel(0).reset_index()

counts = list(ib_data_6h.groupby('SID'))[:3]


test_event = ib_data_6h[ib_data_6h['SID'] =='2000160N21267']

#%%

lat = test_event.LAT.iloc[0]
lon = test_event.LON.iloc[0]

sst_data.sel(time='2000-06-07T18:00:00').sel(latitude=slice(lat+5,lat-5),
                                              longitude=slice(lon-5,lon+5))#.plot()

#%%





#plt.contourf(vals.r.values)

#ax[0].contourf(vals.r.values)


def extrapolate_fields(dset):
    try :
        dset_n = dset.reindex(latitude=list(reversed(dset.latitude)
                                        )).interpolate_na(dim='latitude',
                                                          fill_value="extrapolate"
                                                          ).interpolate_na(
                                                              dim='longitude',
                                                              fill_value="extrapolate")
    
    except ValueError:
        dset_n = dset.interpolate_na(dim='latitude',fill_value="extrapolate"
                                                              ).interpolate_na(
                                                                  dim='longitude',
                                                                  fill_value="extrapolate")
    return dset_n



extrapolate_fields(sst_data.sel(time='2000-06-07T18:00:00')).plot()



#%%
import numpy as np
import cv2
from PIL import Image
vals_all = []
out_file = []
for label,group in counts:
    
    i_ib = group
    print(group['SID'].iloc[0])
    for ii in range(len(i_ib)):
        lat = i_ib['LAT'].iloc[ii]
        lon = i_ib['LON'].iloc[ii]
        time = i_ib['datetime'].iloc[ii]
        lead = i_ib['lead'].iloc[ii]
        try :
            in_data = sh_data.sel(time=time)
        except KeyError:
            print('Time not found')
            continue
        
        
            
        aoi = prepro_features()
        
        
        try:
            rh = aoi.angular_imgs(var_dset=sh_data,lat=lat,lon=lon,box=True)
            rh = extrapolate_fields(rh)
            
        except ValueError:
            continue
        
        for ii in rh.latitude.values:
            if ii in i_ib.LAT:
                pass
            
            else:
                continue
        
        for jj in rh.longitude.values:
            if jj in i_ib.LON:
                pass
            
            else:
                continue
        vo = aoi.angular_imgs(var_dset=vort_data,lat=lat,lon=lon,box=True)
        vo = extrapolate_fields(vo)
        vs = aoi.angular_imgs(var_dset=shear_data,lat=lat,lon=lon,box=True)
        vs = extrapolate_fields(vs)
        sst = aoi.angular_imgs(var_dset=sst_data,lat=lat,lon=lon,box=True)
        sst = extrapolate_fields(sst)
        sp = aoi.angular_imgs(var_dset=pres_data,lat=lat,lon=lon,box=True)
        sp = extrapolate_fields(sp)
        mask = rh.isnull().any()
        
        
            
        feature_set = np.concatenate((rh.values,vo.values,
                                    vs.values,sst.values,sp.values),axis=1)
            
            
        rh_datas += [rh]
        vo_datas += [vo]
        vs_datas += [vs]
        sst_datas += [sst]
        sp_datas += [sp]
            

        out_file += [test_out + label +'_' + str(lead) + '_test.npy']
        #cv2.imwrite(out_file,vals.values)
            
        #np.save(out_file,)
            
for ii in range(len(vals_all)):
    np.save(out_file[ii],vals_all[ii])
    
#%%

n_ibdata = ib_data_6h

def create_feature_set(variable,lat,lon):
    
    
    
    
    aoi = prepro_features()
    var_ang = aoi.angular_imgs(var_dset=variable,lat=lat,lon=lon,box=True)
    
    if (len(pd.unique(var_ang.longitude)) <= 1) | (len(pd.unique(var_ang.latitude)) <= 1):
        return None
    else :
        if bool(var_ang.isnull().any()):
            print("NAN values found")
        
            ext_data = extrapolate_fields(variable)
            n_dset = aoi.angular_imgs(ext_data)
        else: 
            n_dset = var_ang 
            
        return n_dset





test_event = ib_data_6h[ib_data_6h['SID'] =='2000266N12337']

#in_data = sh_data.isel(time=1)
#lat = test_event.LAT.iloc[0]
#lon = test_event.LON.iloc[0]
arrays = []
test_out = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/datasets/images/test/'

vars_all = [sst_data,pres_data,vort_data,shear_data,sh_data]
feat_names = ['sst','pres','vort','shear','rh']
ori_data = sst_data
for jj in range(len(test_event)):
    lat = test_event.LAT.iloc[jj]
    lon = test_event.LON.iloc[jj]
    time = test_event.datetime.iloc[jj]
    all_dsets = []
    for dat in vars_all:
        in_data = dat.sel(time=time)
        dset = create_feature_set(in_data,lat,lon)
        all_dsets += [dset]
    final_dset = xr.concat(all_dsets,dim=feat_names)
    final_dset = final_dset.rename({'concat_dim':'feature'})
    id_name = test_event['SID'].iloc[0]
    lead = test_event['lead'].iloc[jj]
    #arrays += [dset.values]
    lead_str = ("{:03d}".format(int(lead)))
    #print(test_out+id_name+'_'+lead_str+'.nc')
    final_dset.to_netcdf(test_out+id_name+'_'+lead_str+'.nc')

#%%

for var in vars_all:
    in_data = var
    for jj in range(len(test_event)):
        lat = test_event.LAT.iloc[jj]
        lon = test_event.LON.iloc[jj]
        time = test_event.datetime.iloc[jj]
        dset = create_feature_set(in_data,lat,lon)
        