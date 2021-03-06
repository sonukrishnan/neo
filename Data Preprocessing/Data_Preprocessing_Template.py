#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:33:35 2017

@author: sdn
"""

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
"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
"""
#Encoding of Categorical Data
#-----------------------------
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
"""

                
#Splitting the dataset to Training and Test
#-------------------------------------------
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_Train,Y_Test = train_test_split(X,Y, test_size = 0.2,random_state = 0)

#Feature Scaling
#----------------
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


