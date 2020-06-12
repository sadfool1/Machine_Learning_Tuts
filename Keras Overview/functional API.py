"Build complex models"
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


"""
The tf.keras.Sequential model is a simple stack of layers that cannot represent arbitrary models. 
Use the Keras functional API to build complex model topologies such as:

1. Multi-input models,
2. Multi-output models,
3. Models with shared layers (the same layer called several times),
4. Models with non-sequential data flows (e.g. residual connections).

Building a model with the functional API works like this:

1. A layer instance is callable and returns a tensor.
2. Input tensors and output tensors are used to define a tf.keras.Model instance.
3. This model is trained just like the Sequential model.

The following example uses the functional API to build a simple, fully-connected network:
"""
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

inputs = tf.keras.Input(shape=(32,))  # Returns an input placeholder

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)

predictions = layers.Dense(10)(x)

#Instantiate the model given inputs and outputs.

model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)