#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 23:51:44 2023

@author: nmathewa
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%%

#in_fol = '/Volumes/New Volume/Other_works/tropcyc/atlantic_exps2/preprocessing/'
#in_fol = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps2/preprocessing/'
in_fol = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Preprocessing/'


#cifar_data = tf.keras.datasets.cifar10.load_data()

#y_labels = cifar_data[0][1]

#x = cifar_data[0][0]



x_data = np.load(in_fol+'final_arrv3.npy')

y_speeds = pd.read_csv(in_fol+'targetsv3.csv')['USA_WIND'].to_numpy()

plt.plot(y_speeds)

#%%

support_file = pd.read_csv(in_fol+'support_file3.csv').set_index('id')

support_file['seq_no'] = support_file.groupby('id').count()['lead_time']

sequences = pd.unique(support_file['seq_no'])




#%%


support_file = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Preprocessing/support_file3.csv'

dft_sup = pd.read_csv(support_file)


unique_sids = dft_sup['id'].unique()

train_sids, test_sids = train_test_split(unique_sids,test_size=0.2,random_state=1)


train_data = dft_sup[dft_sup['id'].isin(train_sids)]

test_data = dft_sup[dft_sup['id'].isin(test_sids)]

n_data_x,n_data_y = [],[]
for ii in train_data.index:
    n_data_x += [x_data[ii,:,:,:]]
    n_data_y += [y_speeds[ii]]
    
 
x_train = np.array(n_data_x)
y_train = np.array(n_data_y)               



n_data_x,n_data_y = [],[]
for ii in test_data.index:
    n_data_x += [x_data[ii,:,:,:]]
    n_data_y += [y_speeds[ii]]

x_test = np.array(n_data_x)
y_test = np.array(n_data_y)






#%%
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





#%% Sequencing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l1_l2
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,LSTM, Conv3D,ConvLSTM2D

from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import ZeroPadding3D

from tensorflow.keras.preprocessing.sequence import pad_sequences





#%%

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


max_len = 71


# convert tjhe 3d data in to sequences


padded_seq_x = []
padded_seq_y = [] 

for ii in sequences:
    frame_x = norm_x_train[:ii,:,:,:]
    padded_seq_x += [np.pad(frame_x,((0,max_len-ii),(0,0),(0,0),(0,0)))]
    frame_y = norm_y_train[:ii]
    padded_seq_y += [np.pad(frame_y,((0,max_len-ii)))]
    

padded_x_test = []
padded_y_test = []

for ii in sequences:
    frame_x = norm_x_test[:ii,:,:,:]
    padded_x_test += [np.pad(frame_x,((0,max_len-ii),(0,0),(0,0),(0,0)))]
    frame_y = norm_y_test[:ii]
    padded_y_test += [np.pad(frame_y,((0,max_len-ii)))]




final_x_train = np.array(padded_seq_x)
final_y_train = np.array(padded_seq_y)

from tensorflow.keras.layers import Reshape
from tensorflow.keras import Input
from keras.layers import Masking

input_shape = final_x_train.shape



model = Sequential()


input_shape = norm_x_train.shape
model = Sequential()
model.add(Masking(mask_value=0., input_shape=input_sha))
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3)))

#%%

model.compile(loss='mae',optimizer='adam')


#%%


#%%


x_test = np.expand_dims(norm_x_train,axis=0)

y_test = np.expand_dims(norm_y_train,axis=0)

model.fit(x_test,y_test,epochs=10,steps_per_epoch=len(sequences))







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
new_array = np.stack(padded_x_test,axis=0)


targets = model.predict(new_array)

#%%

new_targets = targets*(y_test.max() - y_test.min()) + y_test.min()


new_targets_re = new_targets#.reshape(54*71)

true_targets = np.array(padded_y_test)*((y_test.max() - y_test.min()) + y_test.min())

#%%

test = pd.DataFrame(new_targets_re,columns=['predicted'])
test['true'] = np.array(true_targets)


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
out_dir = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Preprocessing/Results/'
out_model = model.save_weights(out_dir+'no_RNN_tes1')

#%%
model.save(out_dir+'no_RNN_tes1.keras')

#%%

test_new = tf.keras.models.load_model(out_dir+'no_RNN_tes1.keras')


test_new.evaluate(norm_x_train,norm_y_train)

#%%

test_new.predict(norm_x_test)




