# Import Libraries
#-----------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

# Set Seaborn style
sns.set_style("whitegrid")

# Import Dataset
#---------------
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [0,2,3,4,5,6,7,8,9,10,11]].values       
Y = dataset.iloc[:,1].values
                
dataset.head()

### Now let's prepare lists of numeric and categorical columns
# Numeric Features
numeric_features = ['Age', 'Fare']
# Categorical Features
ordinal_features = ['Pclass', 'SibSp', 'Parch']
nominal_features = ['Sex', 'Embarked']

# Check for Missing values
#--------------------------
dataset.isnull().sum()

# Create new column with matched names
#======================================
dataset['target_name'] = dataset['Survived'].map({1:"Survived",0:"Not Surived"})

sns.countplot(dataset.target_name)

### Corralation matrix heatmap
# Getting correlation matrix
cor_matrix = dataset[numeric_features + ordinal_features].corr().round(2)
# Plotting heatmap 
fig = plt.figure(figsize=(12,12));
sns.heatmap(cor_matrix, annot=True, center=0, cmap = sns.diverging_palette(250, 10, as_cmap=True), ax=plt.subplot(111));
plt.show()

### Plotting Numeric Features
# Looping through and Plotting Numeric features
for column in numeric_features:    
    # Figure initiation
    fig = plt.figure(figsize=(18,12))
    
    ### Distribution plot
    sns.distplot(dataset[column].dropna(), ax=plt.subplot(221));
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Density', fontsize=14);
    # Adding Super Title (One for a whole figure)
    plt.suptitle('Plots for '+column, fontsize=18);
    
    ### Distribution per Survived / Not Survived Value
    # Not Survived hist
    sns.distplot(dataset.loc[dataset.Survived==0, column].dropna(),
                 color='red', label='Not Survived', ax=plt.subplot(222));
    # Survived hist
    sns.distplot(dataset.loc[dataset.Survived==1, column].dropna(),
                 color='blue', label='Survived', ax=plt.subplot(222));
    # Adding Legend
    plt.legend(loc='best')
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Density per Survived / Not Survived Value', fontsize=14);
    
    ### Average Column value per Survived / Not Survived Value
    sns.barplot(x="target_name", y=column, data=dataset, ax=plt.subplot(223));
    # X-axis Label
    plt.xlabel('Survived or Not Survived?', fontsize=14);
    # Y-axis Label
    plt.ylabel('Average ' + column, fontsize=14);
    
    ### Boxplot of Column per Survived / Not Survived Value
    sns.boxplot(x="target_name", y=column, data=dataset, ax=plt.subplot(224));
    # X-axis Label
    plt.xlabel('Survived or Not Survived?', fontsize=14);
    # Y-axis Label
    plt.ylabel(column, fontsize=14);
    # Printing Chart
    plt.show()
    
    
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

#=================
#SVM Classification
#=================
"""
rom sklearn.svm import SVC
classifier = SVC(random_state=0, kernel='linear')
classifier.fit(X_Train, y_Train)

"""

#=================
#Kernel SVM Classification
#=================
"""
rom sklearn.svm import SVC
classifier = SVC(random_state=0, kernel='rbf')
classifier.fit(X_Train, y_Train)

"""
"""
#======================================
# Visualising the Training set results
#======================================
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""