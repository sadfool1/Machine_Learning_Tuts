#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:42:10 2020

@author: jameselijah
"""

"""
Custom layers
Create a custom layer by subclassing tf.keras.layers.Layer and implementing the following methods:

1. __init__: Optionally define sublayers to be used by this layer.
2. build: Create the weights of the layer. Add weights with the add_weight method.
3. call: Define the forward pass.
4. Optionally, a layer can be serialized by implementing the get_config method and the from_config class method.

Here's an example of a custom layer that implements a matmul of an input with a kernel matrix:
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

class MyLayer(layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


#Create a model using your custom layer:

model = tf.keras.Sequential([
    MyLayer(10)])

# The compile step specifies the training configuration
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
