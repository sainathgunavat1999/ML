from pandas import DataFrame
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import linear_model

data=pd.read_csv('housing.csv')

column=['Id','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'
,'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
'Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold',
'YrSold','SalePrice']

df = DataFrame(data,columns=column).fillna(value=0)

X = df[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'
,'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
'Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold',
'YrSold']] 
Y = df['SalePrice']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

newMSSubClass=20
newLotFrontage=65
newLotArea=10000
newOverallQual=5
newOverallCond=8
newYearBuilt=2000
newYearRemodAdd=2005
newMasVnrArea=150
newBsmtFinSF1=600
newBsmtFinSF2=0
newBsmtUnfSF=150
newTotalBsmtSF=1200
new1stFlrSF=580
new2ndFlrSF=146
newLowQualFinSF=0
newGrLivArea=1200
newBsmtFullBath=1
newBsmtHalfBath=0
newFullBath=2
newHalfBath=1
BedroomAbvGr=3
KitchenAbvGr=1
TotRmsAbvGrd=8
Fireplaces=1
GarageYrBlt=2000
GarageCars=1
GarageArea=540
WoodDeckSF=500
OpenPorchSF=50
EnclosedPorch=200
3SsnPorch=0
ScreenPorch=0
PoolArea=0
MiscVal=0
MoSold=12
YrSold=2008
print ('Predicted Stock Index Price: \n', regr.predict([['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'
,'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
'Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold',
'YrSold']]))
