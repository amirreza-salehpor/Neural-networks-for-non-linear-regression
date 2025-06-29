
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing


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

Xtrain, Xtest,Ytrain, Ytest=train_test_split(Xscaled, Y,test_size=0.3)

model=linear_model.LinearRegression()
model.fit(Xtrain, Ytrain)

trainprediction = model.predict(Xtrain)
testprediction = model.predict(Xtest)


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

feature_importances=model.coef_
features=['GrlivArea','LotArea','YearBuilt']
for col in Xhousestyle.columns:
    features.append(col)
    
plt.figure()
plt.barh(features,feature_importances[0])

#%%

print('1Story houses average: %.2f'%df['SalePrice'].loc[df['HouseStyle']=='1Story'].mean())
print('2Story houses average: %.2f'%df['SalePrice'].loc[df['HouseStyle']=='2Story'].mean())



