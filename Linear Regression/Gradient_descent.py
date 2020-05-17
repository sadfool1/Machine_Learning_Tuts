# -*- coding: utf-8 -*-
"""
Linear regression - Gradient descent & cost function
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

x = np.array ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y = np.array ([4,5,7,11,13,14,17,20,23,26,29,32,35,38,41])

print (len(x), len (y))


def gradient_des(x,y):
    
    m_innit = b_innit = 0
    iterations = 1000
    learning_rate = 0.011
    n = len(x)
    
    for i in range(iterations):
        data = pd.DataFrame({ "x" : range (1,len(x)+1), "line": m_innit*x + b_innit})

        
        y_prediction = m_innit * x + b_innit #Y = mX + b
        
        cost = (1/n) * sum([value**2 for value in (y-y_prediction)])
        
        m_derivative = -(2/n)*sum(x*(y-y_prediction))
        b_derivative = -(2/n)*sum(y-y_prediction)
        
        m_innit = m_innit - learning_rate * m_derivative
        b_innit = b_innit - learning_rate * b_derivative
        
        
        print ("m: {} \nb: {} \ncost: {} \niteration: {}".format(m_innit,
               b_innit, cost, i), "\n")
        
    
        plt.plot("x", "line", data = data, marker = "", color = "red")
    
    plt.scatter(x,y, color = 'b')
    plt.show()
        
gradient_des(x, y)
        
        