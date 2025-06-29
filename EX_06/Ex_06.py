
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf 

XX=[]

df=pd.read_csv('roasting_data.csv')
X=np.array(df[[
    'T_data_1_1','T_data_1_2','T_data_1_3'
    ,'T_data_2_1','T_data_2_2','T_data_2_3'
    ,'T_data_3_1','T_data_3_2','T_data_3_3'
    ,'T_data_4_1','T_data_4_2','T_data_4_3'
    ,'T_data_5_1','T_data_5_2','T_data_5_3'
    ,'H_data'
    ,'AH_data']])

scaler= preprocessing.StandardScaler()
Xscaled=scaler.fit_transform(X)

Y=np.array(df[['quality']])

scalero=preprocessing.StandardScaler()
yscaled=scalero.fit_transform(Y)

Xtrain, Xtest,Ytrain, Ytest=train_test_split(Xscaled, yscaled,test_size=0.3)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(30,activation=tf.nn.relu,
                          input_shape=(Xtrain.shape[1],)),
    tf.keras.layers.Dropout(0.3),
     tf.keras.layers.Dense(30,activation=tf.nn.tanh),
     tf.keras.layers.Dropout(0.3),
     tf.keras.layers.Dense(30,activation=tf.nn.relu),
     tf.keras.layers.Dense(Ytrain.shape[1])
     ])

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['mae'])

model.fit(Xtrain,Ytrain,validation_data=(Xtest,Ytest), epochs=50,batch_size=10)

trainprediction =scalero.inverse_transform(model.predict(Xtrain))
testprediction =scalero.inverse_transform(model.predict(Xtest))

Xtraincont=scaler.inverse_transform(Xtrain[:,0:17])
Xtrain=np.concatenate((Xtraincont,Xtrain[:,17:]),axis=1)
Xtestcont=scaler.inverse_transform(Xtest[:,0:17])
Xtest=np.concatenate((Xtestcont,Xtest[:,17:]),axis=1)

Ytrain=scalero.inverse_transform(Ytrain)
Ytest=scalero.inverse_transform(Ytest)

'''model=linear_model.LinearRegression()
model.fit(Xtrain, Ytrain)
trainprediction = model.predict(Xtrain)
testprediction = model.predict(Xtest)'''



print('Mean absolute error in train data %.2f'% mean_absolute_error(Ytrain,trainprediction))
print('Mean absolute error in test data %.2f'% mean_absolute_error(Ytest,testprediction))
print('R2 score in train data %.2f'% r2_score(Ytrain,trainprediction))
print('R2 score in test data %.2f'% r2_score(Ytest,testprediction))

