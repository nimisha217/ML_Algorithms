# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:16:23 2020

@author: NI20092396
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)

data = pd.read_csv(r'C:\Users\NI20092396\flask\Salary_Data.csv')

print(data.shape)
data.head()

X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values

meanX = np.mean(X)
meanY = np.mean(Y)
print(meanX)
print(meanY)
m = len(X)

numer = 0
denom = 0

for i in range(m):
    numer += (X[i] - meanX)*(Y[i] - meanY)
    denom += (X[i] - meanX)**2

b1 = numer / denom
b0 = meanY - (b1*meanX)

print(b1, b0)


max_x = np.max(X) 
min_x = np.min(X) 

x = np.linspace(min_x, max_x, 30)
y = b0 + b1*x

plt.plot(x,y, color = '#58b970', label='Regression Line')
plt.scatter(X, Y, color = '#ef5423', label='Scatter Plot')
 
plt.xlabel('salary')
plt.ylabel('experience')
plt.legend()
plt.show()


rmse = 0
for i in range(m):
    y_pred = b0 + b1*X[i]
    rmse += (Y[i] - y_pred)**2
    
rmse = np.sqrt(rmse/m)
print(rmse)


ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - meanY) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)