#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 01:48:36 2018

@author: timo
"""

import tensorflow as tf
from tensorflow import keras

class ConvBaseNet(keras.Model):
    def __init__(self, input_shape, name="base_net"):
        """Base net that is shared by the policy and the value function
        Args:
          policy: 
          name: Name of the module.
        """
        super(ConvBaseNet, self).__init__(name=name)
        self.dense1 = tf.layers.Dense(1)
        self.dense2 = tf.layers.Dense(1)
        self.dense3 = tf.layers.Dense(1)
        self.dense4 = tf.layers.Dense(1)

    def call(self, input):
        out = self.dense1(input)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dense4(out)
        return out


class DenseBaseNet(keras.Model):
    def __init__(self, input_shape, name="base_net"):
        """Base net that is shared by the policy and the value function
        Args:
          policy: 
          name: Name of the module.
        """
        super(DenseBaseNet, self).__init__(name=name)
        self.dense1 = tf.layers.Dense(1)
        self.dense2 = tf.layers.Dense(1)
        self.dense3 = tf.layers.Dense(1)
        self.dense4 = tf.layers.Dense(1)


    def call(self, input):
        out = self.dense1(input)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dense4(out)
        return out
