#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:29:35 2023

@author: nmathewa
"""
import os
os.chdir('/Users/nalex2023/main/tropcyc/modules/')

from process_ib import ib_processor


filename = '/Users/nalex2023/main/tropcyc/atlantic_exps/Datasets/ibtracs.NA.list.v04r00.csv'


ib_proc = ib_processor(csv_loc=filename)

fil_dft = ib_proc.filter_data()

all_dft = ib_proc.compute_cols(data=fil_dft)



all_dft.to_csv("/Users/nalex2023/main/tropcyc/atlantic_exps2/datasets/proc_tracks.csv")