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
X_train,X_test,Y_Train,Y_Test = train_test_split(X,Y, test_size = 0.2,random_state = 0)

# Feature Scaling not applicable as X has only one variable

