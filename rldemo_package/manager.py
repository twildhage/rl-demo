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

# Custom modules
from utils import get_Agent, get_agent, Logger


class AbstractManager(ABC):
    
    @abc.abstractmethod
    def __init__(self, mode, algo, game, render, num_threads, kwargs_agent):
        
        self._mode = mode 
        
        self._algo = algo
        
        self._game = game
        
        self._render = render 
        
        self.num_threads = num_threads
        
        self._kwargs_agent = kwargs_agent
        
        self.log = Logger.get_active_logger()

    @abc.abstractmethod
    def run(self, num_episodes, num_timesteps):
        pass

    @abc.abstractmethod
    def summary(self):
        pass
    
    

