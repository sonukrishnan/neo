# Import Libraries
#-----------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
#---------------
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, :-1].values   
Y = dataset.iloc[:, 1].values

# No Missing Value applicable
                
#Encoding of Categorical Data
#-----------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Dependent Variable Steps
#1 Label Encoder for Dependent Variable X
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])

#2 OneHotEncoder for X to make dummy variables
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap
X = X[:, 1:]
          
#Splitting the dataset to Training and Test
#-------------------------------------------
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.20,random_state = 0)

# Feature Scaling not applicable as X has only one variable

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(X_train,Y_train)

#Predicting the Test set results
Y_Pred = MLR.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
MLR_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
MLR_OLS.summary() ## noticed that P value is high for x2 

X_opt2 = X[:, [0, 1, 3, 4, 5]]
MLR_OLS = sm.OLS(endog = Y, exog = X_opt2).fit()
MLR_OLS.summary() ## noticed that P value is high for x1

X_opt3 = X[:, [0, 3, 4, 5]]
MLR_OLS = sm.OLS(endog = Y, exog = X_opt3).fit()
MLR_OLS.summary() ## noticed that P value is high for x2 

X_opt4 = X[:, [0, 3, 5]]
MLR_OLS = sm.OLS(endog = Y, exog = X_opt4).fit()
MLR_OLS.summary() ## noticed that P value is high for x2 

X_opt5 = X[:, [0, 3]]
MLR_OLS = sm.OLS(endog = Y, exog = X_opt5).fit()
MLR_OLS.summary() 

