# -*- coding: utf-8 -*-
"""
@author: Rishabh Jain
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('steel.csv')

X = dataset.iloc[:, [0,1,3,4,6,7,8,9,10]]
y = dataset.iloc[:, 11].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=3)
# Fit the classifier to the data
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
y_pred
knn.score(X_test, y_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

plot = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=plot)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plot = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=plot)

# 10 fold Cross Validation
from sklearn.model_selection import cross_val_score

#train model with cv of 10 
cv_scores = cross_val_score(knn, X, y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores)
print(np.mean(cv_scores))


#k errors 

from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt

rmse_val = [] #to store rmse values for different k
for K in range(50):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)