#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Nov  4 09:49:02 2023

@author: nmathewa
"""

import os
os.chdir('/Users/nalex2023/main/tropcyc/modules/')
from prepro_features import prepro_features
import xarray as xr
import pandas as pd
import numpy as np
#in_datasets = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'
in_datasets = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Datasets/ERA5_processed_40/'

sh_data = xr.open_dataset(in_datasets+'RH_data40y.nc').r#.isel(time=1)

sst_data = xr.open_dataset(in_datasets+'sst_data_40y.nc').sst.reindex(time=sh_data.time)

shear_data = xr.open_dataset(in_datasets+'shear_data40y.nc').speed_shear.reindex(time=sh_data.time,
                                                                                latitude=sh_data.latitude,
                                                                                longitude=sh_data.longitude)

vort_data = xr.open_dataset(in_datasets+'vort_data_40y.nc')
pres_data = xr.open_dataset(in_datasets+'spres_data_40y.nc').sp.reindex(time=sh_data.time)


#%%



#%%

cor_data =  2 * 7.29 * 1e-5 * np.sin(pres_data['latitude'])

n_data = np.repeat(cor_data.values[:,np.newaxis],len(pres_data['longitude']),axis=1)

n_data_time = np.repeat(n_data[np.newaxis],len(pres_data['time']),axis=0)

cor_dset = xr.DataArray(data=n_data_time,name='cor',
                        dims=['time','latitude','longitude'],
                        coords=dict(
                            latitude=(["latitude"], pres_data.latitude.values),
                            longitude=(["longitude"], pres_data.longitude.values),
                            time = (["time"], pres_data.time.values)))


#%%
main_dir = '/Users/nalex2023/main/tropcyc/atlantic_exps/Datasets/'

in_events = main_dir+'final_proc1980.csv'



ib_data = pd.read_csv(in_events).reset_index()

ib_data['datetime'] = pd.to_datetime(ib_data['datetime'])

#%%
ib_data_6h = ib_data.set_index('datetime').groupby('SID').resample('6H').first().droplevel(0).reset_index()

ib_data_6h = ib_data_6h.set_index('datetime').dropna(axis=0,how='all').reset_index()




#%%
pers_list= []
dist2land = []
lead_list = []
for ii in range(len(ib_data_6h.dropna(axis=1,how='all'))):
    time = ib_data_6h['datetime'].iloc[ii]
    pers = ib_data_6h['pers'].iloc[ii]
    dist2 = ib_data_6h['DIST2LAND'].iloc[ii]
    lead = ib_data_6h['lead'].iloc[ii]
    
    pers_array = np.expand_dims(np.tile(pers,pres_data.shape[1:3]),0)
    lead_array = np.expand_dims(np.tile(lead,pres_data.shape[1:3]),0)
    dist2_array = np.expand_dims(np.tile(dist2,pres_data.shape[1:3]),0)
    
    pers_list += [xr.DataArray(data=pers_array,name='pers',
                               dims=['time','latitude','longitude'],
                               coords=dict(latitude=(["latitude"], pres_data.latitude.values),
                               longitude=(["longitude"], pres_data.longitude.values),
                               time = (["time"],[time])
                                   ))]
    
    dist2land += [xr.DataArray(data=dist2_array,name='dist2land',
                               dims=['time','latitude','longitude'],
                               coords=dict(latitude=(["latitude"], pres_data.latitude.values),
                               longitude=(["longitude"], pres_data.longitude.values),
                               time = (["time"],[time])
                                   ))]
    
    lead_list += [xr.DataArray(data=lead_array,name='lead',
                               dims=['time','latitude','longitude'],
                               coords=dict(latitude=(["latitude"], pres_data.latitude.values),
                               longitude=(["longitude"], pres_data.longitude.values),
                               time = (["time"],[time])
                                   ))]
#%%

ib_test = ib_data_6h.set_index('datetime').dropna(axis=0,how='all')

#%%
pers_data = xr.concat(pers_list,dim='time')
lead_data = xr.concat(lead_list,dim='time')
dist_data = xr.concat(dist2land,dim='time')

#%%



ddd = pers_data.isnull().any()

#%%

#temp_data = xr.open_dataset(in_datasets+'rh_mid_new.nc')


ib_data = pd.read_csv(in_events)

ib_data['datetime'] = pd.to_datetime(ib_data['datetime'])

ib_data_6h = ib_data.set_index('datetime').groupby('SID').resample('6H').first().droplevel(0).reset_index()

ib_data_6h = ib_data_6h.set_index('datetime').dropna(axis=0,how='all').reset_index()

counts = list(ib_data_6h.groupby('SID'))[:3]

 
test_event = ib_data_6h[ib_data_6h['SID'] =='2023264N13336']

#%%

lat = test_event.LAT.iloc[0]
lon = test_event.LON.iloc[0]

sst_data.sel(time='2000-06-07T18:00:00').sel(latitude=slice(lat+5,lat-5),
                                              longitude=slice(lon-5,lon+5)).plot()

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
vals_all = []
vars_all = []
n_ibdata = ib_data_6h

def create_feature_set(variable,lat,lon):
    
    aoi = prepro_features()
    var_ang = aoi.angular_imgs(var_dset=variable,lat=lat,lon=lon,box=True)
    
    if (len(pd.unique(var_ang.longitude)) <= 1) | (len(pd.unique(var_ang.latitude)) <= 1):
        return np.NAN # returning nans
    else :
        if bool(var_ang.isnull().any()):
            print("NAN values found")
        
            ext_data = extrapolate_fields(variable)
            n_dset = aoi.angular_imgs(ext_data,lat=lat,lon=lon,box=True)
        else: 
            n_dset = var_ang 
            
        return n_dset





test_event = ib_data_6h[ib_data_6h['SID'] =='2000266N12337']

#in_data = sh_data.isel(time=1)
#lat = test_event.LAT.iloc[0]
#lon = test_event.LON.iloc[0]
arrays = []
test_out = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/datasets/images/test/'
final_out = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Datasets/Images/'

ori_data = sst_data



for jj in range(len(test_event)):
    lat = test_event.LAT.iloc[jj]
    lon = test_event.LON.iloc[jj]
    time = test_event.datetime.iloc[jj]
    all_dsets = []
    for dat in vars_all:
        in_data = dat.sel(time=time)
        dset = create_feature_set(in_data,lat,lon).to_dataset()
        if dset == np.NAN:
            print("NAN")
            continue
        all_dsets += [dset]
    final_dset = xr.merge(all_dsets)
    #final_dset = final_dset.rename({'concat_dim':'feature'})
    id_name = test_event['SID'].iloc[0]
    lead = test_event['lead'].iloc[jj]
    #arrays += [dset.values]
    lead_str = ("{:03d}".format(int(lead)))
    #print(test_out+id_name+'_'+lead_str+'.nc')
    #final_dset.to_netcdf(test_out+id_name+'_'+lead_str+'.nc')

#%%
import numpy as np


ctr = 0
def create_dsets(event,input_vars,var_names,out_dir):
    # select an single event
    for jj in range(len(event)):
        lat = event.LAT.iloc[jj]
        lon = event.LON.iloc[jj]
        time = event.datetime.iloc[jj]
        pers = event['pers'].iloc[jj]
        dist2 = event['DIST2LAND'].iloc[jj]
        lead = event['lead'].iloc[jj]
        all_dsets = []
        
        for kk in range(len(input_vars)):
            dat = input_vars[kk]
            var = var_names[kk]
            #print(var)
            #print(dat)
            try :
                in_data = dat.sel(time=time)
            except KeyError:
                print('Time not found skipping')
                dset = None
                continue
            try :
                dset = create_feature_set(in_data,lat,lon).to_dataset()
            except AttributeError:
                #dset = np.NAN
                continue
                
            all_dsets += [dset]
        try :
            final_dset = xr.merge(all_dsets)
            
        except TypeError:
            print("skipping out of bound")
            continue
        id_name = event['SID'].iloc[0]
        lead = event['lead'].iloc[jj]
        lead_str = ("{:03d}".format(int(lead)))
        
        try : 
            len(final_dset.latitude)
            
        except AttributeError:
            continue
    
        
        #print(final_dset.r.shape)
        lead_array = np.tile(lead,(40,40))
        dist_array = np.tile(dist2,(40,40))
        pers_array = np.tile(pers,(40,40))
        cor_data =  2 * 7.29 * 1e-5 * np.sin(final_dset['latitude'])
        n_data = np.repeat(cor_data.values[:,np.newaxis],40,axis=1)
        
        
        print(final_dset.r.shape)
        #print(time)
        
        final_dset['cor'] = xr.DataArray(data=n_data,name='corio',
                                   dims=['latitude','longitude'],
                                   coords=dict(latitude=(["latitude"], final_dset.latitude.values),
                                   longitude=(["longitude"], final_dset.longitude.values)
                                       ))
        final_dset['lead'] = xr.DataArray(data=lead_array,name='lead',
                                   dims=['latitude','longitude'],
                                   coords=dict(latitude=(["latitude"], final_dset.latitude.values),
                                   longitude=(["longitude"], final_dset.longitude.values)
                                       ))
        
        final_dset['pers'] = xr.DataArray(data=pers_array,name='pers',
                                   dims=['latitude','longitude'],
                                   coords=dict(latitude=(["latitude"], final_dset.latitude.values),
                                   longitude=(["longitude"], final_dset.longitude.values)
                                       ))
        final_dset['dist'] = xr.DataArray(data=dist_array,name='dist',
                                   dims=['latitude','longitude'],
                                   coords=dict(latitude=(["latitude"], final_dset.latitude.values),
                                   longitude=(["longitude"], final_dset.longitude.values)
                                       ))
        final_dset.to_netcdf(out_dir+id_name+'_'+lead_str+'.nc')

        

vars_all = [sst_data,pres_data,vort_data,shear_data,sh_data]#,cor_dset,dist_data,lead_data, pers_data]
feat_names = ['sst','pres','vort','shear','rh']#,'corio','dist2land','lead','pers']
groups_all = ib_data_6h.groupby('SID')

for label,event in groups_all:
    n_event = event.reset_index()
    create_dsets(n_event,vars_all,feat_names,final_out)
