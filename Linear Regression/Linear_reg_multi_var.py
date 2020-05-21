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
import math

"""
home prices = m1 * area + m2 * bedrooms +  m3 * age + b 

here we have 1 dependent variable and 3 independent variables (a.k.a features)

we have the slopes known as coefficients and b is known as intercept

"""

df = pd.read_csv("Book3 - Multiple variables.csv")

median_bedrooms = math.floor(df.Bedrooms.median())

df.Bedrooms = df.Bedrooms.fillna(median_bedrooms) #fills the NaN --> DATA CLEANING

regression = linear_model.LinearRegression()
regression.fit(df[["Area", "Bedrooms","Age"]],df.Prices)

print (regression.predict([[100000,20,100]]))

#Notes: Linear Regression with multi var introduces extra features in a similar way as single Var
