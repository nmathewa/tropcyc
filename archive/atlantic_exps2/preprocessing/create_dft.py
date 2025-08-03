#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:57:53 2023

@author: nmathewa
"""

import xarray as xr
import pandas as pd
import numpy as np
import glob
import os

#in_files = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/datasets/images/'
#in_files = '/Volumes/New Volume/Other_works/tropcyc/atlantic_exps2/dataset/images/'
#in_files = '/Users/nalex2023/main/tropcyc/atlantic_exps2/datasets/images/'

in_files = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Datasets/Images/'

nc_files = glob.glob(in_files+'*.nc')






nc_dft = pd.DataFrame(nc_files,columns=['files'])

nc_dft['cyclone_id'] = nc_dft['files'].str.split(os.sep).str[-1].str.split('_').str[0]

nc_dft['lead_time'] = nc_dft['files'].str.split(os.sep).str[-1].str.split('_').str[1].str[:3]

#out_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/preprocessing/'

out_dir = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Preprocessing/'


nc_dft.to_csv(out_dir+'final_events_v1_43y.csv',index=False)
