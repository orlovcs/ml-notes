
import pandas as pd
import quandl 
import math
import datetime
import numpy as np #computing library
from sklearn import preprocessing, model_selection, svm # vector machine
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


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
X = preprocessing.scale(X)  #scale data before classifier, scale new values along with training data
#X = X[:-forecast_output+1] #only have X's when have values for ydf.dropna(inplace=True)
#X=X[:-forecast_output]
X_lately=X[-forecast_output:]
df.dropna(inplace=True)
y = np.array(df['label'])
print(len(X), len(y))


X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.2) # take features, labels, shuffles points and outputs
clf =  LinearRegression(n_jobs=20) # switch algorithms, specifies threads 

clf.fit(X_train,y_train) # fit = train

accuracy = clf.score(X_test, y_test) # score = test

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_output)

 #trains and test seperate data because if training classifier to predict data you're testing against the results will be the same

#print(accuracy) # what the accuracy would be predicting the price shift by the end of the day
# accuracy is squared error

# for each algo, look of it can be threaded in the documentation, amount of n_jobs that can be dedicated to it
# linear regression much more threadable

df['Forecast'] = np.nan # empty data
last_date = df.iloc[-1].name
last_unixval = last_date.timestamp()
one_day = 86400
next_unixval = last_unixval + one_day

#x,y don't necessarily correspond to axis, X/features & y/label
# label happens to be the price so y is the correct axis

for i in forecast_set:
   next_date = datetime.datetime.fromtimestamp(next_unixval)
   next_unixval += one_day
   df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + i
   # iterate through forecase set, take each forecast and day and set them as values in data frame, make future features not a number