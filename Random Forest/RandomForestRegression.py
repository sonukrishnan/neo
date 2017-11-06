#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:56:52 2017

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
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)
"""


# Fitting Random Forest Regression to the Training Set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,Y)

# Predicting Value
Y_Pred = regressor.predict(6.5)

# Visualizing the RandomForest Regression
# Grid is created to show minor changes and its values so that the Tree is clearly called out
X_grid = np.arange(min(X), max (X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

import statsmodels.formula.api as sm
sm.OLS(endog=Y, exog=X).fit().summary()

