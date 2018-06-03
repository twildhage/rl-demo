#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:40:32 2018

@author: timo
"""

import argparse

class Config(argparse.Action):
    discount_factor = 0.95
    TENSORBOARD_PATH = '../tensorboard'
    CHECKPOINTS_PATH = '../checkpoints'
    TENSORBOARD_RECORD_INTERVALL = 1
    
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        self.set_cls_var(namespace, values)
        
    @classmethod
    def set_cls_var(cls, var, value):
        setattr(cls, var, value)
        