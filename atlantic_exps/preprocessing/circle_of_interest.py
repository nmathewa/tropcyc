#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 02:04:41 2023

@author: nmathewa
"""

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
pd.options.mode.chained_assignment = None  # default='warn'
main_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'

ibdata = pd.read_csv(main_dir+'final_proc.csv')
ibdata['datetime'] = pd.to_datetime(ibdata['datetime'])

test_data = ibdata[ibdata['SID'] == '2001301N27325']

dft = pd.read_csv(data_dir+'ibtracs.NA.list.v04r00.csv',keep_default_na=False).iloc[1:,:]
dft['datetime'] = pd.to_datetime(dft['ISO_TIME'],format='%Y-%m-%d %H:%M:%S')


#%%

sst_data = xr.open_dataset(main_dir+'sst_new.nc').sst
mask_land = sst_data.notnull()

pres_data = xr.open_dataset(main_dir+'Spres_new.nc').sp.where(mask_land)

rh_data = xr.open_dataset(main_dir+'rh_mid_new.nc').r.where(mask_land)

vort_data = xr.open_dataset(main_dir+'vort_850_new.nc').vo.where(mask_land)

#%%

rh_data.r.isel(time=0)

#%%
def angular_average(var_dset,lat,lon,radius=1000,test=False):
    if isinstance(var_dset,float):
        return var_dset
    
    else :
        latmin = int(lat) - 10
        latmax = int(lat) + 10
        lonmin = int(lon) - 10
        lonmax = int(lon) + 10
        center = (lat,lon)
    
    
        lats = np.arange(latmin,latmax,1)
        lons = np.arange(lonmin,lonmax,1)
    
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
        
        
        masked_var_data = var_dset_sub.where(cond_array)
    
        annul_mean = masked_var_data.mean()
        
        if test:
            return masked_var_data,annul_mean
        else :
            return annul_mean 

#%%

lat,lon = test_data['LAT'].iloc[0],test_data['LON'].iloc[0]
testrh = angular_average(rh_data.isel(time=0),lat,lon,vertical=True)



#%%


groups = ibdata.groupby('SID')


out_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Images/'

test_data = dft[dft['SID'] == '2000266N12337']


fig,ax = plot_map()
for ii in range(len(test_data)):
    time = test_data['datetime'].iloc[ii]
    
    lat = test_data['LAT'].iloc[ii]
    lon = test_data['LON'].iloc[ii]
    sst_data_t = sst_data.sel(time=time)
    pres_data_t = pres_data.sel(time=time)
    amean_pres = angular_average(sst_data_t,lat,lon,test=True)
    amean_pres[0].plot(ax=ax,transform=ccrs.PlateCarree(),add_colorbar=False,
                       cmap='RdBu')
    ax.plot(lon,lat,color='black',marker='.')
ax.set_title('')

fig.savefig(out_dir+"Hurricane ISSAC- Sep 2009",bbox_inches='tight')
#%%

amean_pres[0].plot()
    

#%%


def point_val(data,lat,lon):
    if isinstance(data,float):
        return data
    else :
        data_s = data.sel(latitude=lat,longitude=lon,method='nearest')
    return data_s

#%%
dfts = []
ctr = 0
for event_id , event in groups:
    ctr += 1
    
    n_event = event.sort_values('lead')
    
    print(str(round((ctr/len(groups))*100)))
    
    all_means = []
    all_sst_means = []
    all_mean_rh = []
    all_mean_vo = []
    for ii in range(len(n_event)):
        
        time = n_event['datetime'].iloc[ii]
        
        lat = n_event['LAT'].iloc[ii]
        lon = n_event['LON'].iloc[ii]
        
        try :
            pres_data_t = pres_data.sel(time=time)
            amean_pres = point_val(pres_data_t,lat,lon).values
        except KeyError:
            amean_pres = np.nan
        
        try :
            rh_t = rh_data.sel(time=time)
            amean_rh =  angular_average(rh_t,lat,lon,radius=500).values
        except KeyError:
            amean_rh = np.nan
            
        try :
            sst_t = sst_data.sel(time=time)
            amean_temp = angular_average(sst_t,lat,lon,radius=500).values
        except KeyError:
            amean_temp = np.nan
            
        try :
            vort_t = vort_data.sel(time=time)
            amean_vort = angular_average(vort_t,lat,lon,radius=1000).values

        except KeyError:
            amean_vort = np.nan
        
        
        
       
        
            
        
        all_means += [amean_pres]
        
        all_sst_means += [amean_temp]
        all_mean_rh += [amean_rh]
        all_mean_vo += [amean_vort]
        
    
    n_event['pres_mean'] = all_means
    n_event['sst_mean'] = np.array(all_sst_means)
    n_event['rh_data'] = np.array(all_mean_rh)
    n_event['vo_data'] = np.array(all_mean_vo)
    dfts += [n_event]
    
final_arr = pd.concat(dfts)
#%%
final_arr['cor_param'] = 2 * (7.29e-5) * np.sin(final_arr['LAT'])
#%%

final_arr.to_csv(main_dir+'final3333.csv')

        


#%%

fig,ax = plt.subplots()

final_arr[final_arr['SID'] == '2000231N14308']['pres_mean'].plot(ax=ax)

ax2 = ax.twinx()

final_arr[final_arr['SID'] == '2000231N14308']['USA_WIND'].plot(ax=ax2,color='b')



#%%




def plot_map():
    
    fig,ax = plt.subplots(figsize=(20,10),subplot_kw={"projection": ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)


    ax.coastlines()
    
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='black', alpha=0.1, linestyle='--')
    ax.set_extent([-100, 30, 0, 80], crs=ccrs.PlateCarree())
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    #data.plot(ax=ax,transform=ccrs.PlateCarree())

    return fig,ax

#%%
final_arr.to_csv(main_dir+'test1111.csv')

