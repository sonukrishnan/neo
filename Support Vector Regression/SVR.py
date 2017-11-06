#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:44:42 2017

@author: sdn
"""

# Import Libraries
#-----------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
#---------------
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values   
Y = dataset.iloc[:, 2].values

# No Missing Value applicable

# Encoding of Categorical Data not required
          
#Splitting the dataset to Training and Testnot applicable as its small

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)



# Fitting Support Vector Regression to the Training Set
from sklearn.svm import SVR
svrmodel = SVR(kernel='rbf')
svrmodel.fit(X,Y)

# Predicting Simple
#Scale the value of 6.5 to Standard Scaler
# Transform is expecting an array and hence we need np.array function
X1 = sc_X.transform(np.array([[6.5]]))

# Since we are now working on transformed data, we need to do inverse transform
sc_Y.inverse_transform(svrmodel.predict(X1))


# Visualizing the SVR Regression

plt.scatter(X, Y, color = 'red')
plt.plot(X, svrmodel.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
"""

# Visualizing the Polynomial Linear Regression
#X_grid = np.arange(min(X), max(X), 0.1) #..This is if you want to see continous line
#X_grid = X_grid.reshape(len(X_grid),1) #.. reshape is for changing to array
plt.scatter(X, Y, color = 'red')
#plt.plot(X_grid, PLR.predict(polynomialfeatures.fit_transform(X_grid)), color = 'green')
plt.plot(X, PLR.predict(polynomialfeatures.fit_transform(X)), color = 'blue')
plt.title('Polynominal Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Predict PLR
PLR.predict(polynomialfeatures.fit_transform(6.5))




