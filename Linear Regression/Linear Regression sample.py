#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:10:01 2020

@author: jameselijah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("Book1.csv")

plt.scatter(df.Area, df.Price, color = 'b')
plt.xlabel ("area")
plt.ylabel ("price (U$SD)")
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[["Area"]], df[["Price"]])

y_predicted = reg.predict([[3300]])

intercept = reg.intercept_
slope = reg.coef_

y_val = slope* 3300 + intercept

if y_predicted == y_val:
    print (True)
    
d = pd.read_csv("book2.csv")
