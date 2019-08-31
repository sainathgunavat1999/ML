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
newBedroomAbvGr=3
newKitchenAbvGr=1
newTotRmsAbvGrd=8
newFireplaces=1
newGarageYrBlt=2000
newGarageCars=1
newGarageArea=540
newWoodDeckSF=500
newOpenPorchSF=50
newEnclosedPorch=200
new3SsnPorch=0
newScreenPorch=0
newPoolArea=0
newMiscVal=0
newMoSold=12
newYrSold=2008
print ('Predicted House Price: \n', regr.predict([[newMSSubClass,newLotFrontage,newLotArea,newOverallQual,newOverallCond,newYearBuilt,newYearRemodAdd,newMasVnrArea,newBsmtFinSF1,newBsmtFinSF2,newBsmtUnfSF
,newTotalBsmtSF,new1stFlrSF,new2ndFlrSF,newLowQualFinSF,newGrLivArea,newBsmtFullBath,newBsmtHalfBath,newFullBath,newHalfBath,newBedroomAbvGr,newKitchenAbvGr,newTotRmsAbvGrd,newFireplaces,newGarageYrBlt,newGarageCars,newGarageArea,newWoodDeckSF,newOpenPorchSF,newEnclosedPorch,new3SsnPorch,newScreenPorch,newPoolArea,newMiscVal,newMoSold,
newYrSold]]))
