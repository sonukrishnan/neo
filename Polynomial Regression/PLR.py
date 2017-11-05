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
#-----------------------------
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Dependent Variable Steps
#1 Label Encoder for Dependent Variable X
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

#2 OneHotEncoder for X to make dummy variables
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap
X = X[:, 1:]
          
#Splitting the dataset to Training and Test
#-------------------------------------------
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.20,random_state = 0)
"""
# Feature Scaling not applicable as X has only one variable

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
PLR = Pol








