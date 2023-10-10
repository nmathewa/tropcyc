#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:48:16 2023

@author: nmathewa
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten


dataset = pd.read_csv(main_dir+'test_modelin1.csv')

features = dataset[['lead','pers','temp_mean',
                    'pres_mean','DIST2LAND']].values

target = dataset['USA_WIND'].values

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(len(dataset), activation='relu',input_shape=(5,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(model.fit(features,target))


#%%

model.fit(features,target,epochs=100)

