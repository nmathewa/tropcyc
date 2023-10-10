#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:48:16 2023

@author: nmathewa
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.preprocessing import StandardScaler 


dataset = pd.read_csv(main_dir+'test_modelin1.csv').dropna(how='any',axis=0)





scaler = StandardScaler()  

df_Scaled = scaler.fit_transform(dataset[['lead','pers','temp_mean',
                    'pres_mean','DIST2LAND','USA_WIND']])



x_train = dataset[['pers','temp_mean',
                    'pres_mean']]#df_Scaled[:,:-1]



y_train = dataset['USA_WIND']

x_train_r = x_train.values.reshape(x_train.shape[0],3)
y_train_r = y_train.values.reshape(x_train.shape[0])

#%%

n_xtrain = np.array(x_train_r)
n_ytrain = np.reshape(y_train_r, (len(y_train_r), 1))


#%%

def set_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1028,input_shape=(3,),activation='relu'))
    #model.add(tf.keras.layers.Dense(128,activation='sigmoid'))
    #hidden layers 
    model.add(tf.keras.layers.Dense(512,activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    # the model is compiled with the loss function and the optimizer
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


model = set_model()
model.fit(n_xtrain,n_ytrain,epochs=100)

