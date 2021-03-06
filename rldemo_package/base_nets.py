#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    MIT License
#    
#    Copyright (c) 2018 Timo Wildhage
#    
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

import tensorflow as tf
from tensorflow import keras

class ConvBaseNet(keras.Model):
    def __init__(self, input_shape, name="base_net"):
        """Convolutional base net that are the same for the policy and the value
        function. Therefore, sharing the weights is also possible.
        Args:
          policy: 
          name: Name of the module.
        """
        super(ConvBaseNet, self).__init__(name=name)
        
        self.conv1 = keras.layers.Conv2D(24,
                                        (7,7),
                                        strides=(4, 4),
                                        padding='valid',
                                        data_format=None,
                                        dilation_rate=(1, 1),
                                        activation=keras.activations.relu,
                                        use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros',
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None)
        self.conv2 = keras.layers.Conv2D(24,
                                        (7,7),
                                        strides=(4, 4),
                                        padding='valid',
                                        data_format=None,
                                        dilation_rate=(1, 1),
                                        activation=keras.activations.relu,
                                        use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros',
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None)
        self.conv3 = keras.layers.Conv2D(24,
                                        (7,7),
                                        strides=(4, 4),
                                        padding='valid',
                                        data_format=None,
                                        dilation_rate=(1, 1),
                                        activation=keras.activations.relu,
                                        use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros',
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None)
        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(0.5)
        
        
    def call(self, input):
        
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        out = self.dropout(out)
        return out


class DenseBaseNet(keras.Model):
    def __init__(self, input_shape, name="base_net"):
        """Dense base net that are the same for the policy and the value
        function. Therefore, sharing the weights is also possible.
        Args:
          policy: 
          name: Name of the module.
        """
        super(DenseBaseNet, self).__init__(name=name)
        self.dense1 = keras.layers.Dense(100, input_dim=input_shape)
        self.dense2 = tf.layers.Dense(10)

    def call(self, input):
        out = self.dense1(input)
        out = self.dense2(out)
        return out
