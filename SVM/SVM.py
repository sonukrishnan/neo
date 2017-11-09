#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 23:04:58 2017

@author: sdn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
#---------------
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:-1].values   
y = dataset.iloc[:, 4].values

# Dataset Split
#--------------
from sklearn.cross_validation import train_test_split
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
#----------------
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

#SVM Classification
#-------------
from sklearn.svm import SVC
classifier = SVC(random_state=0, kernel='linear')
classifier.fit(X_Train, y_Train)

#Prediction
#----------
y_Pred = classifier.predict(X_Test)

# Confusion Matrix
#-----------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_Test, y_Pred)




 