#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:48:16 2023

@author: nmathewa
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.preprocessing import StandardScaler 
import pandas as pd


main_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'
 


dataset = pd.read_csv(main_dir+'test_modelin1.csv').dropna(how='any',axis=0)


dataset_sort = dataset.sort_values('SID')

#%%



scaler = StandardScaler()  

df_Scaled = dataset_sort[['lead','pers','temp_mean',
                    'pres_mean','DIST2LAND','USA_WIND']]



x_train = dataset_sort[['pers','temp_mean',
                    'pres_mean','lead']].values#df_Scaled[:,:-1]



y_train = dataset['USA_WIND'].values




# Define the model
model = Sequential([
    Dense(1028, activation='relu', input_shape=(4,)),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=20)

# Make predictions
y_pred = model.predict(x_train)
#%%

dddd = pd.DataFrame(y_pred)

dddd['original'] = y_train
