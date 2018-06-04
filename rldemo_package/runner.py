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
    def run(self, coord=None, num_episodes='default', num_timesteps='default'):
        pass

    @abc.abstractmethod
    def train(self, coord=None, num_episodes='default', num_timesteps='default'):
        pass

    @abc.abstractmethod
    def eval(self, coord=None, num_episodes='default', num_timesteps='default'):
        pass

    def thread(self, coord, mode, num_episodes='default', num_timesteps='default'):
        
        return Thread(target=getattr(self, mode),
                      kwargs={'coord': coord,
                              'num_episodes':num_episodes,
                              'num_timesteps':num_timesteps})


