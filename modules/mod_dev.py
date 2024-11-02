#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:46:56 2024

@author: nmathewa
"""

try:
    import cupy as np
    print('using cupy')
except ImportError:
    import numpy as np

import pandas as pd


import os 

os.chdir('/home/nmathewa/main/tropcyc/modules/')

out_test = '/media/nmathewa/nma_backup/Datasets/Other_works/tropcyc/atlantic_exps3/preprocessing/final_arrv3.npy'

out_data = np.load(out_test)

sup_file = '/media/nmathewa/nma_backup/Datasets/Other_works/tropcyc/atlantic_exps3/preprocessing/support_file3.csv'

sup_dft = pd.read_csv(sup_file)


tar_file = '/media/nmathewa/nma_backup/Datasets/Other_works/tropcyc/atlantic_exps3/preprocessing/targetsv3.csv'

tar_df = pd.read_csv(tar_file)


#%%


from process_ib import ib_processor


ib_data = '/media/nmathewa/nma_backup/Datasets/Other_works/tropcyc/atlantic_exps/Datasets/ibtracs.NA.list.v04r00.csv'

processor = ib_processor(csv_loc=ib_data).compute_cols()







