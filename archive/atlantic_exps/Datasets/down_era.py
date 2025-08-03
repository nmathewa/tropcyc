#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:53:31 2023

@author: nmathewa
"""

import cdsapi

#vort = ['vorticity',850]

#hum = ['relative_humidity',[300,325,350,375,400,425,450,475,500]]

uwind = ['u_component_of_wind', [200,250,300,775,800,925]]


c = cdsapi.Client()

for ii in range(2000,2024):
    year = str(ii)
    print(year)
    
    #c.retrieve('reanalysis-era5-pressure-levels',
    c.retrieve('reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable':['u_component_of_wind', 'v_component_of_wind']
        ,
        'pressure_level': uwind[1],
        'year': year,
        'month': [str(jj).zfill(2) for jj in range(6,12)],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '03:00',
            '06:00',
            '09:00', 
            '12:00', 
            '15:00',
            '18:00', 
            '21:00'
        ],
        'area': [
            43, -83, 2,
            -5,
        ],
    },
    year+"_UVWIND_surface"+'.nc')