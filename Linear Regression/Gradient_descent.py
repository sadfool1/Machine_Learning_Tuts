# -*- coding: utf-8 -*-
"""
Linear regression - Gradient descent & cost function
"""

import numpy as np
import random

x = np.array ([1,2,3,4,5])
y = np.array ([4,5,10,11,16])

def gradient_des(x,y):
    m_innit = b_innit = 0
    iterations = 1000
    learning_rate = 0.004
    n = len(x)
    
    for i in range(iterations):
        
        y_prediction = m_innit * x + b_innit #Y = mX + b
        m_derivative = -(2/n)*sum(x*(y-y_prediction))
        b_derivative = -(2/n)*sum(y-y_prediction)
        
        m_innit = m_innit - learning_rate * m_derivative
        b_innit = b_innit - learning_rate * b_derivative
        
        print ("m: {} \nb: {} \niteration: {}".format(m_innit,
               b_innit, i), "\n")
        
gradient_des(x, y)
        
        