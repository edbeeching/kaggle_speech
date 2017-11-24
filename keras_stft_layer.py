# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:24:57 2017

@author: Edward
"""

from keras import backend as K
from keras.engine.topology import Layer
from tf.contrib.signal import stft

class Spec(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Spec, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(Spec, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return stft(x, self.kernel)

    def compute_output_shape(self, input_shape):
        
        
        return (input_shape[0], self.output_dim)