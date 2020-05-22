#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:34:43 2020

@author: jameselijah
"""


import tensorflow as tf
#tensor2 = tf.reshape([])

"""
with tf.Session as sess:
    tf.eval()
"""

t = tf.ones([3,3,3,3])

#print(t)

t = tf.reshape(t, [27])

print (t)
