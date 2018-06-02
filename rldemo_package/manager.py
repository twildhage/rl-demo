#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:01:49 2018

@author: timo
"""

# Build-in modules
from abc import ABC
import abc
from threading import Thread

# Third party modules 
import gym

# Custom modules
from utils import get_Agent, get_agent, Logger


class AbstractManager(ABC):
    
    @abc.abstractmethod
    def __init__(self, mode, algo, game, kwargs_agent):
        
        self._algo = algo
        
        self._game = game
        
        self._mode = mode 
        
        self._kwargs_agent = kwargs_agent
        
        self.log = Logger.get_active_logger()

    @abc.abstractmethod
    def run(self, num_episodes, num_timesteps):
        pass

    @abc.abstractmethod
    def summary(self):
        pass
    
    

