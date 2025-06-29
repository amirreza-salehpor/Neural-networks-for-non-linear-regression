
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf 


df=pd.read_csv('HousePrice.csv')
X=np.array(df[['GrLivArea','LotArea','YearBuilt']])

scaler= preprocessing.StandardScaler()
Xscaled=scaler.fit_transform(X)
Xhousestyle=pd.get_dummies(df['HouseStyle'])

for col in Xhousestyle.columns:
    if len(Xhousestyle[Xhousestyle[col]==1])<20:
        Xhousestyle=Xhousestyle.drop(col,axis=1)

Xscaled= np.concatenate((Xscaled,Xhousestyle), axis=1)
Y=np.array(df[['SalePrice']])

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

Xtraincont=scaler.inverse_transform(Xtrain[:,0:3])
Xtrain=np.concatenate((Xtraincont,Xtrain[:,3:]),axis=1)
Xtestcont=scaler.inverse_transform(Xtest[:,0:3])
Xtest=np.concatenate((Xtestcont,Xtest[:,3:]),axis=1)

Ytrain=scalero.inverse_transform(Ytrain)
Ytest=scalero.inverse_transform(Ytest)


print('Mean absolute error in train data %.2f'% mean_absolute_error(Ytrain,trainprediction))
print('Mean absolute error in test data %.2f'% mean_absolute_error(Ytest,testprediction))

print('R2 score in train data %.2f'% r2_score(Ytrain,trainprediction))
print('R2 score in test data %.2f'% r2_score(Ytest,testprediction))

plt.figure()
plt.scatter(Xtrain[:,0], Ytrain, label='Actual prices')
plt.scatter(Xtrain[:,0], trainprediction, label='predicted prices')
plt.title('Train data prediction')
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.legend()

plt.figure()
plt.scatter(Xtest[:,0], Ytest, label='Actual prices')
plt.scatter(Xtest[:,0], testprediction, label='predicted prices')
plt.title('test data prediction')
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.legend()

#feature_importances=model.coef_
#features=['GrlivArea','LotArea','YearBuilt']
#or col in Xhousestyle.columns:
    #features.append(col)
    
#plt.figure()
#plt.barh(features,feature_importances[0])





