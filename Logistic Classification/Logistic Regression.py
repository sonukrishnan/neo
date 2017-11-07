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
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values   
Y = dataset.iloc[:, 4].values
