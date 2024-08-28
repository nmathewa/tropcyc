#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:01:27 2024

@author: nalex2023
"""


def plot_with_area2(mjo_dft):

    mjo_dft_n = mjo_dft.drop('day0',axis=1).rolling('5D').mean()

    fig,ax = plt.subplots(1,1,figsize=(20,12),subplot_kw={'projection':ccrs.PlateCarree(
        central_longitude=130)})
    ax.coastlines()
    

    """
    #label_style = {'size': 20, 'color': 'black', 'weight': 'bold'}
    #gl.xlabel_style = label_style
    #gl.ylabel_style = label_style
    #ax.stock_img()
    #ax.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    #ax.set_yticks([ -15, 0, 15], crs=ccrs.PlateCarree())
    ax.yaxis.tick_right()
    ax.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(draw_labels=True,linewidth=1,color='gray',alpha=0.5,linestyle='--')
    #set the size of the title
    
    #ax.add_feature(cfeature.OCEAN,zorder=3,color='white')
    """
    
    

    
    #ax.background_img(name='BM', resolution='high')
    ax.stock_img()
    
    
    lat = mjo_dft_n['lat'].dropna().values
    lon = mjo_dft_n['lon'].dropna().values
    
    
    cmap = plt.cm.get_cmap('jet',len(times))
    norm = plt.Normalize(vmin=0,vmax=len(times))
    #cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,label='Date',shrink=0.3)
    
    colors=['green','orange','r','blue']
    
    phases = [-36,-16,-6,6,15]
    for ii in range(len(colors)):
        
        mjo_phase = mjo_dft_n[(mjo_dft_n['day'] > phases[ii]) &
                              (mjo_dft_n['day'] < phases[ii+1])]
        
        lat = mjo_phase.lat.values
        lon = mjo_phase.lon.values
        
        ax.plot(lon,lat,transform=ccrs.PlateCarree(),linewidth=4,color=colors[ii],zorder=5)
      
        
    
    #ax.plot(lon[0],lat[0],transform=ccrs.PlateCarree(),marker='o',markersize=3,color='b',label='Point of origin')
    #ax.plot(lon[-1],lat[-1],transform=ccrs.PlateCarree(),marker='x',markersize=5,color='r',label='Point of termination')
    ax.plot(manus_lon,manus_lat,transform=ccrs.PlateCarree(),marker='o',markersize=8,color='white')
    ax.text(manus_lon+1.2,manus_lat-1,'Manus',transform=ccrs.PlateCarree(),fontsize=20,color='white',
            weight='bold',path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    
    #ax.plot(nauru_lon,nauru_lat,transform=ccrs.PlateCarree(),marker='o',markersize=8,color='white')
    #ax.text(nauru_lon+1.2,nauru_lat-1,'Nauru',transform=ccrs.PlateCarree(),fontsize=20,color='white',
            #weight='bold',path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    
    pap_lon = 143.6
    pap_lat = -5.4



    ax.text(pap_lon-5,pap_lat,r"Papua New",transform=ccrs.PlateCarree(),fontsize=12,color='white',
            weight = 'bold',zorder=5,path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    ax.text(pap_lon-5,pap_lat-2,r"Guinea",transform=ccrs.PlateCarree(),fontsize=12,
            color='white',weight = 'bold',zorder=5,path_effects=[pe.withStroke(linewidth=4, foreground="black")])

    
    ax.plot([manus_lon,manus_lon],[-20,20],transform=ccrs.PlateCarree(),color='red',
            linewidth=2,linestyle='--',zorder=3)
    
    #ax.plot(darwin_lon,darwin_lat,transform=ccrs.PlateCarree(),marker='o',markersize=8,color='white')
    #ax.text(darwin_lon+1.2,darwin_lat-1,'Darwin',transform=ccrs.PlateCarree(),fontsize=20,color='white',
            #weight='bold',path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    
    

    sum_lon = 100
    sum_lat = 0.7

    ax.text(sum_lon-2.6,sum_lat-2,r"Sumatra",transform=ccrs.PlateCarree(),fontsize=18,
        color='white',weight = 'bold',zorder=5,path_effects=[pe.withStroke(linewidth=4, foreground="black")])


    bor_lon = 114.86
    bor_lat = 1.3

    ax.text(bor_lon-4,bor_lat-1,r"Borneo",transform=ccrs.PlateCarree(),fontsize=18,
        color='white',weight = 'bold',zorder=5,path_effects=[pe.withStroke(linewidth=4, foreground="black")])


    aus_lon = 130
    aus_lat = -19

    ax.text(aus_lon,aus_lat,r"Austraila",transform=ccrs.PlateCarree(),fontsize=18,
        color='white',weight = 'bold',zorder=5,path_effects=[pe.withStroke(linewidth=4, foreground="black")])

    
    ax.text(-70
            ,23,'a)',fontsize=30,weight='bold')
    
    """
    ax.set_xticks([ 80, 100, 120, 140, 160, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_extent([60,200,-20,20],crs=ccrs.PlateCarree())
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='grey', alpha=0.2, linestyle='--', draw_labels=True)
    gl.xlabels_top = True
    gl.ylabels_left = True
    gl.ylabels_right=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator([80,100,120, 140, 160, 180,])
    gl.ylocator = mticker.FixedLocator([-15,-10,-5,0, 5, 10, 15])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 'weight': 'bold','size':20}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold','size':20}
    ax.set_extent([60,200,-20,20],crs=ccrs.PlateCarree())
    n_legend = 2
    #handles,labels = ax.get_legend_handles_labels()
    # plot the name as text
    #ax.text(lon[0],lat[0],str(name),transform=ccrs.PlateCarree(),fontsize=12,color='white')
    return fig,ax




#%%

import requests
import pandas as pd



filename = '/home/nalex2023@fit.edu/main/doe_mjo_arm/works/shinto_pallav_ENSO/lpt_systems_imerg_2021060100_2022063023.txt'


dft = pd.read_csv(filename, sep='\s+', header=1)

dft_all = dft.rename(columns={'YYYYMMDDHH':'datetime',
                                             '_A_[km2]':'Area',
                                             'cen_lat.__':'lat',
                                             'cen_lon.__':'lon',})



headers = []
for ii in range(len(dft_all)):
    if dft_all['datetime'][ii] == 'LPT':
        print('header found at index:', ii)
        headers.append(ii)


# split by location of headers

dft_all_list = []
for ii in range(len(headers)):
    if ii == len(headers)-1:
        dft_all_list.append(dft_all[headers[ii]:])
    else:
        dft_all_list.append(dft_all[headers[ii]:headers[ii+1]])
        
    
final_ind_frames = []
for jj in dft_all_list:
    dft_ind = jj
    dft_ind['lptid'] = dft_ind['Area'].iloc[0]
    dft_ind = dft_ind.dropna(how='any',axis=0)
    final_ind_frames += [dft_ind]
    

#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


dft_all = pd.concat(final_ind_frames)


dft_dd = (dft_all[dft_all['lptid'] == 39.1]).sort_values('datetime')

dft_dd['datetime'] = pd.to_datetime(dft_dd['datetime'],format='%Y%m%d%H')










fig,ax = plt.subplots(1,1,figsize=(20,12),subplot_kw={'projection':ccrs.PlateCarree()})
ax.coastlines()

lat = dft_dd['lat']
lon = dft_dd['lon']

ax.plot(lon,lat,transform=ccrs.PlateCarree(),linewidth=4,zorder=5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='grey', alpha=0.2, linestyle='--', draw_labels=True)

