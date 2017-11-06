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

# Feature Scaling not applicable as X has only one variable

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
SLR = LinearRegression()
SLR.fit(X,Y)

# Fitting Polynominal Regression
from sklearn.preprocessing import PolynomialFeatures
polynomialfeatures = PolynomialFeatures(degree=4)
X_Poly = polynomialfeatures.fit_transform(X)
polynomialfeatures.fit(X_Poly,Y)

PLR = LinearRegression()
PLR.fit(X_Poly,Y)

# Visualizing the Linear Regression
"""
plt.scatter(X, Y, color = 'red')
plt.plot(X, SLR.predict(X), color = 'blue')
plt.title('Simple Linear Regression')
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

# Predicting Simple
SLR.predict(6.5)

# Predict PLR
PLR.predict(polynomialfeatures.fit_transform(6.5))




