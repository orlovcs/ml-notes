
import pandas as pd
import quandl 
import math
import numpy as np #computing library
from sklearn import preprocessing, model_selection, svm # vector machine
from sklearn.linear_model import LinearRegression


quandl.ApiConfig.api_key = "9xfxcdQFN8z3rzEx89LH"
df = quandl.get('WIKI/GOOGL') #df for data frame
df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]      #split df into arrays, 
#high & low tells about volatilty for the day
#open vs close - how much it went up or down
df['HighLowPercent'] = ( df['Adj. High'] - df['Adj. Close'] ) / df['Adj. Close'] * 100
df['ChangePercent'] = ( df['Adj. Close'] - df['Adj. Open'] ) / df['Adj. Open'] * 100
df = df[['Adj. Close','HighLowPercent','ChangePercent','Adj. Volume']] ## volume is amount of trades that day
#print(df.head())

forecast_column = 'Adj. Close' #want future values
df.fillna('-99999', inplace=True) #cannot work with unavailable data, will be foreced to be treated as an outlier

forecast_output = int(math.ceil(0.1 * len(df))) #attempting to forecast 10% of the df 

df['label'] = df[forecast_column].shift(-forecast_output) # we shift the data back by 10% of df to be able to forecast the future values
#attempt to predict adjusted close price
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)  #scale data before classifier, scale new values along with training data
#X = X[:-forecast_output+1] #only have X's when have values for ydf.dropna(inplace=True)
#df.dropna(inplace=True)
y = np.array(df['label'])

#print(len(X), len(y))


X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.2) # take features, labels, shuffles points and outputs
clf =  LinearRegression()

clf.fit(X_train,y_train) # fit = train

accuracy = clf.score(X_test, y_test) # score = test

 #trains and test seperate data because if training classifier to predict data you're testing against the results will be the same

print(accuracy) # what the accuracy would be predicting the price shift by the end of the day
# accuracy is squared error

