from utils import *

import time
import numpy as np
import pandas as pd

from tqdm import tqdm

from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
import datetime
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime

import warnings
warnings.filterwarnings("ignore")

context = mx.cpu(); model_ctx=mx.cpu()
mx.random.seed(1719)

def parser(x):
    return datetime.datetime.strptime(x,'%Y-%m-%d')

data_FT = pd.read_csv('C:/Users/tuege/Downloads/GS (1).csv', header=0, parse_dates=[0], date_parser=parser)

series = data_FT['Close']
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
#plt.figure(figsize=(10, 7), dpi=80)
#plt.show()

from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    print(t, "/", round(len(X)-(len(X)*0.66)), 0)
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

plt.figure()#figsize=(12, 6), dpi=100)
plt.plot(test[400:], label='Real')
plt.plot(predictions[400:], color='red', label='Predicted')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 5: ARIMA model on GS stock')
plt.legend()
plt.show()

print(predictions)

