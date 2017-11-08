# Import Libraries
#-----------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
#---------------
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values       
Y = dataset.iloc[:,3].values
                
#Taking care of Missing values
#-----------------------------
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding of Categorical Data
#-----------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Dependent Variable Steps
#1 Label Encoder for Dependent Variable X
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

#2 OneHotEncoder for X to make dummy variables
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
#Label Encoder for Dependent Variable Y
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Avoiding Dummy Variable Trap
X = X[:, 1:]


#Splitting the dataset to Training and Test
#-------------------------------------------
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_Train,Y_Test = train_test_split(X,Y, test_size = 0.2,random_state = 0)

#Feature Scaling
#----------------
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Regression
#==========
# SLR
#==========
"""
# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
SLR = LinearRegression()
SLR.fit(X_train,Y_Train)

#Predicting the Test set results
Y_Pred = SLR.predict(X_test)

#Visualizing the Training Set Results
plt.scatter(X_train, Y_Train, color = 'red')
plt.plot(X_train, SLR.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test Set Results
plt.scatter(X_test, Y_Pred, color = 'red')
plt.scatter(X_test, Y_Test, color = 'blue')

plt.plot(X_train, SLR.predict(X_train), color='green')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

################
# F Regression #
################

from sklearn.feature_selection import f_regression
freg = f_regression(X_train, Y_Train)

"""

#=========
# MLR
#=========
"""
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

"""

#=======================
# Polynomial Regression
#=======================

"""
# Fitting Polynominal Regression
from sklearn.preprocessing import PolynomialFeatures
polynomialfeatures = PolynomialFeatures(degree=4)
X_Poly = polynomialfeatures.fit_transform(X)
polynomialfeatures.fit(X_Poly,Y)

PLR = LinearRegression()
PLR.fit(X_Poly,Y)
"""

#=================
#  SVR Regression
#=================
"""
# Fitting Support Vector Regression to the Training Set
from sklearn.svm import SVR
svrmodel = SVR(kernel='rbf')
svrmodel.fit(X,Y)

# Predicting SVR (below is only if scale is used)
#Scale the value of 6.5 to Standard Scaler
# Transform is expecting an array and hence we need np.array function
X1 = sc_X.transform(np.array([[6.5]]))

# Since we are now working on transformed data, we need to do inverse transform
sc_Y.inverse_transform(svrmodel.predict(X1))

"""


#===============
# Decision Tree
#===============

"""
# Fitting Decision Tree Regression to the Training Set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

# Predicting Value
Y_Pred = regressor.predict(6.5)

# Visualizing the DTR Regression
# Grid is created to show minor changes and its values so that the Tree is clearly called out
X_grid = np.arange(min(X), max (X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (DTR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
"""


#===============
# Random Forest
#===============

"""

# Fitting Decision Tree Regression to the Training Set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 100, random_state=0)
regressor.fit(X,Y)

# Predicting Value
Y_Pred = regressor.predict(6.5)

# Visualizing the DTR Regression
# Grid is created to show minor changes and its values so that the Tree is clearly called out
X_grid = np.arange(min(X), max (X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (DTR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
"""

#=================
# Classifiers
#=================
"""
#Classification
#-------------
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_Train, y_Train)

#Prediction
#----------
y_Pred = classifier.predict(X_Test)

# Confusion Matrix # To compare Predicted and Actual
#-----------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_Test, y_Pred)

"""

#=================
#KNN Classification
#=================
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier.fit(X_Train, y_Train)
"""