# Import Libraries
#-----------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
#---------------
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values       
Y = dataset.iloc[:, 1].values
                
#Encoding (OHE or LE) not applicable as both variables are Ordinal

#Splitting the dataset to Training and Test
#-------------------------------------------
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_Train,Y_Test = train_test_split(X,Y, test_size = 1/3,random_state = 0)

# Feature Scaling not applicable as X has only one variable

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
