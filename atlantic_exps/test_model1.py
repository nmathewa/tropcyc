#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:48:16 2023

@author: nmathewa
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
main_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'
 

dataset_ori = pd.read_csv(main_dir+'final3333.csv')


dataset = dataset_ori.dropna(how='any',axis=0).drop(columns=['Unnamed: 0.1',
                                                                               'Unnamed: 0',
                                                                               'datetime',
                                                                               'SID',
                                                                               'scale'])


scaler = MinMaxScaler()


df_scaled = pd.DataFrame(scaler.fit_transform(dataset),columns=dataset.columns)

df_scaled['SID'] = dataset_ori['SID']


#%%

x_test = df_scaled[df_scaled['SID'] == '2000266N12337'][['lead', 'pers', 'pres_mean',
       'sst_mean', 'rh_data', 'vo_data', 'cor_param']]

y_test = df_scaled[df_scaled['SID'] == '2000266N12337']['USA_WIND']

#%%


import seaborn as sbs

fig,ax = plt.subplots()
ax2 = ax.twinx()
test['rh_data'].plot(ax=ax)
test['USA_WIND'].plot(ax=ax2,color='r')


#%%

x_train = df_scaled[['lead', 'pers', 'pres_mean',
       'sst_mean', 'rh_data', 'vo_data', 'cor_param']].values


y_train = df_scaled['USA_WIND'].values

#%%
scaler2 = MinMaxScaler()
scaler2.fit(dataset_test['USA_WIND'].values.reshape(-1,1))
y_train = dataset['USA_WIND'].values

def set_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(2056,input_shape=(7,),activation='relu'))
    
    model.add(tf.keras.layers.Dense(1024, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    return model



# Define the model
"""
model = Sequential([
    Dense(2048, input_shape=(7,)),
    Dense(1024,activation='relu'),
    Dense(512,activation='relu'),
    Dense(128,activation='relu'),
    Dense(1,activation='linear')
])
"""

model = set_model()
# Compile the model
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=100)
#%%
# Make predictions

y_pred = model.predict(x_train)


#%%

dddd = pd.DataFrame(y_pred)

dddd['original'] = y_train

dddd['mae'] = abs(dddd[0] - dddd['original'])

dddd.mae.plot()
#%%
def plot_model_hist(model,test=False):

    """
    Will plot the model history for the loss and accuracy
    """
    fig,ax = plt.subplots(figsize=(5,5))
    # calling the keras history object to get the loss and accuracy
    ax.plot(model.history['loss'],label='loss')
    ax.set_title('Loss and Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    sax = ax.twinx()
    
    sax.plot(model.history['accuracy'],color='r',label='accuracy')
    sax.set_ylabel('Accuracy')
    fig.suptitle('Training')
    fig.legend()
    ax.grid()
    # if the test set is given add the validation loss and accuracy
    if test:
        fig,ax = plt.subplots(figsize=(5,5))
        ax.plot(model.history['val_loss'],label='loss')
        ax.set_title('Loss and Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        sax = ax.twinx()
        sax.plot(model.history.history['val_accuracy'],color='r',label='accuracy')
        sax.set_ylabel('Accuracy')
        fig.suptitle('Validation')
        fig.legend()
        ax.grid()


plot_model_hist(history)


