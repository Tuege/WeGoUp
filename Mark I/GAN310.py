from utils import *

import time
import numpy as np
import pandas as pd

from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
import datetime
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import PCA

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")
#%%
context = mx.cpu(); model_ctx=mx.cpu()
mx.random.seed(1719)
#%%
def parser(x):
    return datetime.datetime.strptime(x,'%Y-%m-%d')
#%%
dataset_ex_df = pd.read_csv('C:/Users/tuege/Downloads/GS.csv', header=0, parse_dates=[0], date_parser=parser)
#%%

dataset_ex_df[['Date', 'Close']].head(3)

print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0]))

plt.figure(figsize=(14, 5), dpi=100)
plt.plot(dataset_ex_df['Date'], dataset_ex_df['Close'], label='Goldman Sachs stock')
plt.vlines(datetime.date(2016,4, 20), 0, 270, linestyles='--', colors='gray', label='Train/Test data cut-off')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Figure 2: Goldman Sachs stock price')
plt.legend()
plt.show()