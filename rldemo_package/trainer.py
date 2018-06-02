#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:50:21 2018

@author: timo
"""

# Build-in modules
from abc import ABC
import abc

# Custom modules
from runner import AbstractRunner
from utils import get_agent 



class AbstractTrainer(AbstractRunner):

    @abc.abstractmethod
    def __init__(self, *args):
        
        super(AbstractTrainer, self).__init__(*args)

#    @abc.abstractmethod
#    def loss_op(self, target):
#        pass
#
#
#class BaseTrainer(AbstractTrainer):
#
#    def __init__(self, *args):
#        
#        super(AbstractTrainer, self).__init__(*args)
#
#    def loss_op(self, target):
#        pass
