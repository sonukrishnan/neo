#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 18:52:18 2017

@author: sdn
"""

import pandas as pd
import quandl

df = quandl.get("NSE/INFY")




df['HL_PCT']=(df['High'] - df['Low'])/df['Close'] * 100
#  
df['PCT_Change']=(df['Close'] - df['Open'])/df['Open'] * 100

df['quantity'] = df['Total Trade Quantity']
  
df1 = df[['Close','HL_PCT','PCT_Change','quantity']]

print(df1[len(df1)]);

