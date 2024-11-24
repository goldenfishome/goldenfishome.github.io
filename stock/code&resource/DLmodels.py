#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
np.random.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
from keras import models
from keras import layers
from keras import optimizers
import os
import warnings
warnings.filterwarnings('ignore')


# In[63]:


# Step 1 : import data
data=pd.read_csv('data/stock.csv')
print(data.shape)
data.head()
data = data[['AZN']]
data.head()


# In[64]:


# Step 2 : split data
data = np.array(data.values.astype('float32'))
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data).flatten()
n = len(data)
# Point for splitting data into train and test
split = int(n * 0.8)
train_data = data[range(split)]
test_data = data[split:]


# In[114]:


# Step 3 : Prepare the input X and target Y
def get_XY(dat, time_steps):
    # Indices of target array
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    # Prepare X
    rows_x = len(Y)
    X = dat[range(time_steps * rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))
    return X, Y
time_steps = 12
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)
model = input("choose between 3 model types (RNN, LSTM, GRU): ")
if model == "RNN":
## This is about Keras SimpleRNN:
    def create_model(hidden_units, dense_units, input_shape, activation):
        model = Sequential()
        model.add(SimpleRNN(hidden_units, input_shape=input_shape,
                        activation=activation[0]))
        model.add(Dense(units=dense_units, activation=activation[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
elif model == "LSTM":
    def create_model(hidden_units, dense_units, input_shape, activation):
        model = Sequential()
        model.add(LSTM(hidden_units, input_shape=input_shape,
                        activation=activation[0]))
        model.add(Dense(units=dense_units, activation=activation[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
else:
    def create_model(hidden_units, dense_units, input_shape, activation):
        model = Sequential()
        model.add(GRU(hidden_units, input_shape=input_shape,
                        activation=activation[0]))
        model.add(Dense(units=dense_units, activation=activation[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


# In[115]:


## Step 4: Create RNN Model And Train
## reuse the function: creat_RNN()
model = create_model(hidden_units=3, dense_units=1, input_shape=(time_steps,1),
                   activation=['tanh', 'tanh'])
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)


# In[116]:


## Step 5: Compute And Print The Root Mean Square Error
## use the function: print_error()
def print_error(trainY, testY, train_predict, test_predict):
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))


# In[117]:


# make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)
# Mean square error of Training dataset and Testing dataset:
print_error(trainY, testY, train_predict, test_predict)


# In[118]:


## Step 6: View The result
# Plot the result
def plot_result(trainY, testY, train_predict, test_predict): ## define the content of the plot
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Adjust close Price for AZN Stock Price')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
plot_result(trainY, testY, train_predict, test_predict)
plt.show() ## show the plot


# In[83]:


data=pd.read_csv('data/azn.csv')
data.head()


# In[84]:



## set training and testing dataset
global lag     #forcast time lag
lag=12

nrow=data.shape[0]
train_index=list(range(int(0.7*(nrow-lag))))
validation_index=list(range(int(0.7*(nrow-lag)),int(0.9*(nrow-lag))))
test_index=list(range(int(0.9*(nrow-lag)),(nrow-lag)))

def generate_X_y(data,ex_rate):
    tmp=data[ex_rate]
    print('Raw data mean:',np.mean(tmp),'\nRaw data std:',np.std(tmp))
    tmp=(tmp-np.mean(tmp))/np.std(tmp)

    X=np.zeros((nrow-lag,lag))
    for i in range(nrow-lag):X[i,:lag]=tmp.iloc[i:i+lag]
    y=np.array(tmp[lag:]).reshape((-1,1))
    return (X,y)


# In[85]:


### CHECK train data of X and Y
X,y=generate_X_y(data,'AZN.Close')
X_train,y_train=X[train_index,:],y[train_index,:]
X_validation,y_validation=X[validation_index,:],y[validation_index,:]
X_test,y_test=X[test_index,:],y[test_index,:]


# In[86]:


## get the valiedate and test benchmark value
## define drift validate benchmark:
### CHECK with another method:
def training_performance(model, training_history, epochs):
    test_MAE = np.mean(np.abs(y_test - model.predict(X_test.reshape(-1, lag, 1))))
    timestep = range(1, epochs + 1)
    plt.figure(figsize=(10, 8), facecolor='white')
    plt.subplot(2, 1, 1)
    plt.plot(timestep, np.log(training_history.history['val_mae']), 'b', label='Validation Mean Absolute Error')
    plt.plot(timestep, np.log(training_history.history['mae']), 'bo', label='Training Mean Absolute Error')
    plt.hlines(np.log(drift_validate_benchmark), xmin=timestep[0], xmax=timestep[-1], colors='coral',
               label='Validation Drift Benchmark')
    plt.hlines(np.log(mean_validate_benchmark), xmin=timestep[0], xmax=timestep[-1], colors='lightblue',
               label='Validation Mean Benchmark')
    plt.hlines(np.log(test_MAE), xmin=timestep[0], xmax=timestep[-1], colors='purple', label='Testing Mean Absolute Error')
    plt.ylabel('logged Mean Absolute Error')
    plt.xlabel('Epoch Value')
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.hlines(test_MAE, xmin=timestep[0], xmax=timestep[-1], colors='purple', label='Testing Mean Absolute Error')
    plt.hlines(drift_test_benchmark, xmin=timestep[0], xmax=timestep[-1], colors='coral', label='Test Drift Benchmark')
    plt.hlines(mean_test_benchmark, xmin=timestep[0], xmax=timestep[-1], colors='lightblue',
               label='Test Mean Benchmark')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='right')
    plt.show()

def get_Mae_benchmark(X,y):
    mean_benchmark=np.mean(np.abs(np.mean(X,0)-y))
    drift_benchmark=np.mean(np.abs(X[:,-1]-y))
    return(mean_benchmark,drift_benchmark)

mean_validate_benchmark,drift_validate_benchmark=get_Mae_benchmark(X_validation,y_validation)
mean_test_benchmark,drift_test_benchmark=get_Mae_benchmark(X_test,y_test)



def plot_prediction(model, ex_rate):
    tmp = data[ex_rate]
    plt.figure(figsize=(10, 8), facecolor='white')
    prediction = model.predict(X_test.reshape(-1, lag, 1))
    prediction = prediction * np.std(tmp) + np.mean(tmp)
    y_true = y_test * np.std(tmp) + np.mean(tmp)
    plt.plot(list(range(len(prediction))), prediction, color='coral', label='Prediction')
    plt.plot(list(range(len(y_true))), y_true, color='purple', label='True Value')
    xticks = np.arange(0, len(y_true), 7)
    plt.xticks(xticks, labels=data.Date.iloc[test_index].iloc[xticks], rotation=90)
    plt.legend()
    plt.show()


# In[93]:


### Simple RNN:
optimizer=optimizers.RMSprop()
model_RNN=models.Sequential()
model_RNN.add(layers.SimpleRNN(32,input_shape=(lag,1),activation='relu'))
model_RNN.add(layers.Dense(1))

model_RNN.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

Rnn_history=model_RNN.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=18,epochs=82,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_RNN,Rnn_history,82)
plot_prediction(model_RNN,'AZN.Close')


# In[91]:


### LSTM:
model_LSTM=models.Sequential()
model_LSTM.add(layers.LSTM(32,input_shape=(lag,1),activation='relu'))
model_LSTM.add(layers.Dense(1))
model_LSTM.compile(optimizer=optimizers.RMSprop(),loss='mse',metrics=['mae'])

history_LSTM=model_LSTM.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=18,epochs=82,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_LSTM,history_LSTM,82)
plot_prediction(model_LSTM,'AZN.Close')


# In[90]:


### GRU:
model_GRU=models.Sequential()
model_GRU.add(layers.GRU(32,input_shape=(lag,1),activation='relu'))
model_GRU.add(layers.Dense(1))
model_GRU.compile(optimizer=optimizers.RMSprop(),loss='mse',metrics=['mae'])

history_GRU=model_GRU.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=18,epochs=82,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_GRU,history_GRU,82)
plot_prediction(model_GRU,'AZN.Close')


# In[87]:


### apply regulation on RNN:
### L1 regulation:
optimizer=optimizers.RMSprop()
model_RNN_L1=models.Sequential()
model_RNN_L1.add(layers.SimpleRNN(32, activation='relu', input_shape=(lag,1)))
model_RNN_L1.add(layers.Dense(1, activation='linear', kernel_regularizer='l1'))

model_RNN_L1.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

history_RNN_L1=model_RNN_L1.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=18,epochs=82,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_RNN_L1,history_RNN_L1,82)
plot_prediction(model_RNN_L1,'AZN.Close')


# In[106]:


print('training RMSE:', math.sqrt(np.square(history_RNN_L1.history['loss']).mean()))
print('testing RMSE:', math.sqrt(np.square(history_RNN_L1.history['val_loss']).mean()))


# In[56]:


## apply L2 to RNN:
optimizer=optimizers.RMSprop()
model_RNN_L2=models.Sequential()
model_RNN_L2.add(layers.SimpleRNN(32, activation='relu', input_shape=(lag,1)))
model_RNN_L2.add(layers.Dense(1, activation='linear', kernel_regularizer='l2'))
model_RNN_L2.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

history_RNN_L2=model_RNN_L2.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=18,epochs=82,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_RNN_L1,history_RNN_L2,82)
plot_prediction(model_RNN_L2,'AZN.Close')


# In[107]:


print('training RMSE:', math.sqrt(np.square(history_RNN_L2.history['loss']).mean()))
print('testing RMSE:', math.sqrt(np.square(history_RNN_L2.history['val_loss']).mean()))


# In[ ]:




