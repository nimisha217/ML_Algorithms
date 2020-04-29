# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:40:20 2020

@author: NI20092396
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv(r'C:\Users\NI20092396\flask\Salary_Data.csv')

X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split

Xtrain, xtest, Ytrain, ytest = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(Xtrain, Ytrain)

ypred = regressor.predict(xtest)

plt.scatter(Xtrain, Ytrain, color = 'red')
plt.plot(Xtrain, regressor.predict(Xtrain), color = 'blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show