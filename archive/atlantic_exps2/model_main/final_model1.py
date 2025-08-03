#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:16:03 2023

@author: nmathewa
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#%%

#in_fol = '/Volumes/New Volume/Other_works/tropcyc/atlantic_exps2/preprocessing/'
#in_fol = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/preprocessing/'
in_fol = '/media/nmathewa/nma_backup/Datasets/Other_works/tropcyc/atlantic_exps2/preprocessing/'

#cifar_data = tf.keras.datasets.cifar10.load_data()

#y_labels = cifar_data[0][1]

#x = cifar_data[0][0]



x_data = np.load(in_fol+'final_arrv3.npy')

y_speeds = pd.read_csv(in_fol+'targetsv3.csv')['USA_WIND'].to_numpy()


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

max_vals = []
min_vals = []

for jj in range(x_test.shape[-1]):
    max_vals += [x_test[:,:,:,jj].max()]
    min_vals += [x_test[:,:,:,jj].min()]


norm_x_test = (x_test - min_vals)/(np.array(max_vals) - np.array(min_vals))

#norm_y_train = (y_train - y_train.mean())/y_train.std()

norm_y_test = (y_test - y_test.min())/(y_test.max() - y_test.min())

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

regularizer = tf.keras.regularizers.l2(0.1)

model.add(Conv2D(filters = 32, kernel_size = (1,1),padding = 'Same', 
                 activation ='relu', input_shape = (40,40,9)))
model.add(MaxPool2D(pool_size=(4,4)))
#model.add(Dropout(0.25))
#

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
#model.add(MaxPool2D(pool_size=(4,4)))
#
model.add(Dropout(0.25))
# fully connected

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
#model.add(MaxPool2D(pool_size=(4,4)))


#model.add(Dropout(0.3))



model.add(Conv2D(filters = 32, kernel_size = (1,1),padding = 'Same', 
                 activation ='relu',kernel_regularizer=regularizer))
model.add(MaxPool2D(pool_size=(2,2)))

#model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(32, activation = "relu",))
model.add(Dropout(0.1))
model.add(Dense(1, activation = "relu"))



callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=4)

model.compile(optimizer='adam' , loss = "mae", metrics=["accuracy"])        
model.fit(norm_x_train,norm_y_train,epochs=100,callbacks=[callback])

#%%


def plot_model_hist(model,test=False):

    """
    Will plot the model history for the loss and accuracy
    """
    fig,ax = plt.subplots(figsize=(12,12))
    # calling the keras history object to get the loss and accuracy
    ax.plot(model.history.history['loss'],label='loss',linewidth=5)
    ax.set_title('Loss (MAE)',fontsize=30)
    ax.set_ylabel('Loss',fontsize=30)
    ax.set_xlabel('Epochs',fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=25)

    
    
    return fig,ax 


import seaborn as sbs
sbs.set_theme()
plot_model_hist(model)


#%%
targets = model.predict(norm_x_test)

#%%

new_targets = targets*(y_test.max() - y_test.min()) + y_test.min()


#%%

test = pd.DataFrame(new_targets,columns=['predicted'])
test['true'] = y_test

test.corr()


fig,ax = plt.subplots(figsize=(12,12))
sbs.regplot(test,x='predicted',y='true',ax=ax)


import numpy as np
mae_mean = abs(test['true'] - test['predicted'])
import scipy as sp

r,p = sp.stats.pearsonr(test['predicted'], test['true'])

ax.text(.9, 0.05, 'r={:.2f}'.format(r),
            transform=ax.transAxes,fontsize=20)

ax.set_ylabel('Observed wind speeds (kt)',fontsize=30)
ax.set_xlabel('Predicted wind speeds (kt)',fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=25)


#%%


fig,ax = plt.subplots()

ax.plot(y_test)
ax.plot(new_targets)

##83.42