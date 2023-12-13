#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:43:51 2023

@author: nalex2023
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


support_file = '/Users/nalex2023/main/tropcyc/atlantic_exps3/Preprocessing/support_file4.csv'

dft_sup = pd.read_csv(support_file)


unique_sids = dft_sup['id'].unique()

train_sids, test_sids = train_test_split(unique_sids,test_size=0.2,random_state=1)


train_data = dft_sup[dft_sup['id'].isin(train_sids)]

seq_train = train_data.groupby('id').count()['lead_time'].values


test_data = dft_sup[dft_sup['id'].isin(test_sids)]

seq_test = test_data.groupby('id').count()['lead_time'].values


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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,LSTM, Conv3D,ConvLSTM1D

from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import ZeroPadding3D

from tensorflow.keras.preprocessing.sequence import pad_sequences



#%%

norm_x_trainX = np.array(norm_x_train).reshape(4533,1,40,40,9)
norm_x_testX = np.array(norm_x_test).reshape(1025,1,40,40,9)


model = Sequential()
model.add(TimeDistributed(Conv2D(32,(1,1),activation='relu'),input_shape=(1,40,40,9)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(4,4))))

model.add(TimeDistributed(Conv2D(128,(1,1),activation='relu')))
#model_mix_shoulder.add(TimeDistributed(Conv2D(128,(3,3),activation='relu')))
#model_mix_shoulder.add(TimeDistributed(Conv2D(56,(3,3),activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
model.add(Dropout(0.25))


model.add(TimeDistributed(Conv2D(64,(3,3),activation='relu')))

#model_mix_shoulder.add(TimeDistributed(Conv2D(256,(3,3),activation='relu')))
#model_mix_shoulder.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))


model.add(TimeDistributed(Flatten()))

#RNN
model.add(LSTM(1024,return_sequences=False))

model.add(Dense(1,activation='relu'))
#model.add(activation('sigmoid'))

#model.summary()

#%%

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=4)

model.compile(optimizer='adam' , loss = "mae", metrics=["accuracy"])        
model.fit(norm_x_trainX,norm_y_train,epochs=100,callbacks=[callback])


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

targets = model.predict(norm_x_testX)

plt.plot(targets)

#%%




#%%



new_test = targets*(y_test.max() - y_test.min()) + y_test.min()
#%%




#%%



#%%
new_targets = targets*(y_test.max() - y_test.min()) + y_test.min()




true_targets = np.array(norm_y_test)*((y_test.max() - y_test.min()) + y_test.min())

#%%

test = pd.DataFrame(new_targets,columns=['predicted'])
test['true'] = np.array(true_targets)


test = test#[test['true'] != 0]

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



