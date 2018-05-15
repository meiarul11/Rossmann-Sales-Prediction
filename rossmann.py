# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:52:11 2018

@author: meiar
"""

import pandas as pd
import numpy as np
from sklearn import cross_validation

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
store = pd.read_csv('store.csv')

train.fillna(0, inplace=True)
train.loc[train.Open.isnull(), 'Open'] = 1

train['SchoolHoliday'] = train['SchoolHoliday'].astype(float) 

train['year'] = train.Date.apply(lambda x: x.split('-')[0])
train['year'] = train['year'].astype(float)
train['month'] = train.Date.apply(lambda x: x.split('-')[1])
train['month'] = train['month'].astype(float)
train['day'] = train.Date.apply(lambda x: x.split('-')[2])
train['day'] = train['day'].astype(float)

test['year'] = test.Date.apply(lambda x: x.split('-')[0])
test['year'] = test['year'].astype(float)
test['month'] = test.Date.apply(lambda x: x.split('-')[1])
test['month'] = test['month'].astype(float)
test['day'] = test.Date.apply(lambda x: x.split('-')[2])
test['day'] = test['day'].astype(float)

store.loc[store['StoreType'] == 'a', 'StoreType'] = '1'
store.loc[store['StoreType'] == 'b', 'StoreType'] = '2'
store.loc[store['StoreType'] == 'c', 'StoreType'] = '3'
store.loc[store['StoreType'] == 'd', 'StoreType'] = '4'
store['StoreType'] = store['StoreType'].astype(float)

store.loc[store['Assortment'] == 'a', 'Assortment'] = '1'
store.loc[store['Assortment'] == 'b', 'Assortment'] = '2'
store.loc[store['Assortment'] == 'c', 'Assortment'] = '3'
store['Assortment'] = store['Assortment'].astype(float)

test.fillna(1, inplace=True)

train = train[train["Open"] != 0]

train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }
num_trees = 300

val_size = 100000

X_train, X_test = cross_validation.train_test_split(train, test_size=0.01)

import xgboost as xgb

dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
dtest = xgb.DMatrix(test[features])
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)

train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
print('error', error)

