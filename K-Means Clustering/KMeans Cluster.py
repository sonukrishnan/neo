#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:50:22 2018

@author: sdn
"""

# Import Libraries
#-----------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
#---------------
dataset = pd.read_csv('Mall_Customers.csv')

# Randomize the X inputs prior to split to Train and Test
#========================================================
X = dataset.iloc[:, [3,4]].values       

#Clustering
#========== 
from sklearn.cluster import KMeans
wcss = []

for i in range[1,11]:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    