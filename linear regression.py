# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:21:14 2019

@author: Rishabh Jain
"""

import numpy as np
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('steel.csv')
X = dataset.iloc[:, [0,1,3,4,6,7,8,9,10]]
y = dataset.iloc[:, 11].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_pred
lr.score(X_test, y_test)

plot = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=plot)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.model_selection import cross_val_score
#train model with cv of 10 
score = cross_val_score(lr, X, y, cv=10)
#print each cv score (accuracy) and average them
print(score)
print(np.mean(score))

