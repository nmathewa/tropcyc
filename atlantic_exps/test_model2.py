#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:11:02 2023

@author: nmathewa
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
main_dir = '/home/nmathewa/main/GIT/tropcyc/atlantic_exps/Datasets/'
from sklearn.model_selection import train_test_split 


dataset_ori = pd.read_csv(main_dir+'final3333.csv').sort_values('datetime')


dataset = dataset_ori.dropna(how='any',axis=0).drop(columns=['Unnamed: 0.1',
                                                                               'Unnamed: 0',
                                                                               'datetime',
                                                                               'SID',
                                                                               'scale'])
def plot_model_hist(model,test=False):

    """
    Will plot the model history for the loss and accuracy
    """
    fig,ax = plt.subplots(figsize=(5,5))
    # calling the keras history object to get the loss and accuracy
    ax.plot(model.history.history['loss'],label='loss')
    ax.set_title('Loss (MAE)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    
    
    return fig,ax 

scalerx = MinMaxScaler()
scalery = MinMaxScaler()


scalery.fit(dataset['USA_WIND'].values.reshape(-1,1))


df_scaled = pd.DataFrame(scalerx.fit_transform(dataset),columns=dataset.columns)





df_scaled['SID'] = dataset_ori['SID']


req_cols = ['SID','lead','pres_mean','sst_mean','rh_data',
                        'vo_data','cor_param',
                        'pers','USA_WIND']
n_data = df_scaled[req_cols].dropna(how='any',axis=0)

unique_sids = n_data['SID'].unique()

train_sids, test_sids = train_test_split(unique_sids,test_size=0.2,random_state=1)

train_data = n_data[n_data['SID'].isin(train_sids)]

test_data = n_data[n_data['SID'].isin(test_sids)]



X_train = train_data.drop(['USA_WIND','SID'],axis=1).values

y_train = train_data['USA_WIND'].values


X_test = test_data.drop(['USA_WIND','SID'],axis=1).values
y_test = test_data['USA_WIND'].values
#%%

def set_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024,input_shape=(7,),activation='relu'))
    
    model.add(tf.keras.layers.Dense(512, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    return model

model = set_model()
# Compile the model


model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5)

# Train the model
model.fit(X_train, y_train, epochs=100,callbacks=[callback])

#%%

fig,ax = plot_model_hist(model)

fig.savefig(out_dir + 'final_run.png')
#%%

y_pred = scalery.inverse_transform(model.predict(X_test))

#%%

predictions = pd.DataFrame(y_pred,columns=['predicted'])

predictions['observed'] = scalery.inverse_transform(y_test.reshape(-1,1))




import seaborn as sbs

fig,ax = plt.subplots(figsize=(12,9))
sbs.regplot(predictions,x='predicted',y='observed',ax=ax)

fig.savefig(out_dir+'final_regression.png')
#%%
import numpy as np
mae_mean = abs(predictions['observed'] - predictions['predicted'])
import scipy as sp

r,p = sp.stats.pearsonr(predictions['predicted'], predictions['observed'])


#%%


predictions['SID'] = test_data["SID"].values

#%%

grp_events = predictions.groupby('SID')

dfts = []
for id_name,event in grp_events:
    print(event)
    print(len(event))
    t_len = len(event)
    dfts += [event.sort_values('observed')]
    
    
final_predicts = pd.concat(dfts)
#%%



grp_events = final_predicts.groupby('SID')


for id_name,event in grp_events:
    event['Lead Time (hours)'] = np.arange(0,len(event)*3,3)
    event.set_index('Lead Time (hours)')
    fig,ax = plt.subplots(figsize=(18,10))
    sbs.pointplot(event,x='Lead Time (hours)',y='observed',ax=ax)
    sbs.pointplot(event,x='Lead Time (hours)',y='predicted',ax=ax,color='red')
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel("Lead Time (hours)",fontsize=20)
    fig.suptitle(id_name)
    fig.savefig(out_dir+str(id_name)+'tests.png')
    
#%%




