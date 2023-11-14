#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:09:52 2023

@author: nmathewa
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#%%
in_fol = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/preprocessing/'


cifar_data = tf.keras.datasets.cifar10.load_data()

#y_labels = cifar_data[0][1]

#x = cifar_data[0][0]



x_data = np.load(in_fol+'final_arr.npy')

y_speeds = pd.read_csv(in_fol+'targets.csv')['USA_WIND'].to_numpy()

#%%

x_train, x_test, y_train, y_test = train_test_split(x_data, y_speeds, test_size=0.2, random_state=1)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

#train_y_new = scaler.fit_transform(y_train)


import matplotlib.pyplot as plt


max_vals = []
min_vals = []

for ii in range(x_train.shape[-1]):
    max_vals += [x_train[:,:,:,ii].max()]
    min_vals += [x_train[:,:,:,ii].min()]


norm_x_train = (x_train - min_vals)/(np.array(max_vals) - np.array(min_vals))

#norm_y_train = (y_train - y_train.mean())/y_train.std()

norm_y_train = (y_train - y_train.min())/(y_train.max() - y_train.min())

#%%

#%%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l1_l2
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (10,10,5)))
model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
#model.add(Dropout(0.25))
# fully connected

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "relu"))

from keras.optimizers import RMSprop,Adam
#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer='adam' , loss = "mae", metrics=["accuracy"])        
model.fit(norm_x_train,norm_y_train,epochs=10)

#%%

targets = model.predict(norm_x_train)
#%%

new_targets = targets*(y_train.max() - y_train.min()) + y_train.min()


#%%

test = pd.DataFrame(new_targets,columns=['predicted'])
test['true'] = y_train

test.corr()


#%%


fig,ax = plt.subplots()

ax.plot(y_train)
ax.plot(new_targets)

