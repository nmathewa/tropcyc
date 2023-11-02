#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:29:35 2023

@author: nmathewa
"""
import os
os.chdir('/home/nmathewa/main/GIT/tropcyc/modules/')

from process_ib import ib_processor


filename = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/ibtracs.NA.list.v04r00.csv'


ib_proc = ib_processor(csv_loc=filename)

fil_dft = ib_proc.filter_data()

all_dft = ib_proc.compute_cols(data=fil_dft)



