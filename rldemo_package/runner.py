#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:50:21 2018

@author: timo
"""

# Build-in modules
from abc import ABC
import abc
from threading import Thread

# Third party modules 
import gym

# Custum modules
import base_nets 
from utils import get_Agent, Logger, map_game_to_net


class AbstractRunner(ABC):

    @abc.abstractmethod
    def __init__(self, algo, game, kwargs_agent):
        
        self._algo = algo
        
        self._game = game
        
        self._kwargs_agent = kwargs_agent
        
        self.log = Logger.get_active_logger()

    @abc.abstractmethod
    def run(self, coord=None, num_episodes=199, num_timesteps=50):
        pass

    @abc.abstractmethod
    def train(self, coord=None, num_episodes=199, num_timesteps=50):
        pass

    @abc.abstractmethod
    def eval(self, coord=None, num_episodes=199, num_timesteps=50):
        pass

    def thread(self, coord, mode, num_episodes=199, num_timesteps=50):
        
        return Thread(target=getattr(self, mode), kwargs={'coord': coord})


