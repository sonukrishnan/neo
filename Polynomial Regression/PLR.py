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

from sklearn.preprocessing import PolynomialFeatures
polynomialfeatures = PolynomialFeatures(degree=2)
X_Poly = polynomialfeatures.fit_transform(X)

PLR = LinearRegression()
PLR.fit(X_Poly,Y)

# Visualizing the Linear Regression
plt.scatter(X, Y, color = 'red')
plt.plot(X, SLR.predict(X), color = 'blue')
plt.title('Simple Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Linear Regression
plt.scatter(X, Y, color = 'red')
plt.plot(X, PLR.predict(polynomialfeatures.fit_transform(X)), color = 'green')
plt.title('Polynominal Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



