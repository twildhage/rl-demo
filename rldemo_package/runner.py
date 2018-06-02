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

#    @abc.abstractmethod
#    def run(self, num_episodes=199, num_timesteps=50):
#        pass

    @abc.abstractmethod
    def train(self, coord=None, num_episodes=199, num_timesteps=50):
        pass

    @abc.abstractmethod
    def eval(self, coord=None, num_episodes=199, num_timesteps=50):
        pass

    def thread(self, coord, mode, num_episodes=199, num_timesteps=50):
        
        return Thread(target=getattr(self, mode), kwargs={'coord': coord})



#class BaseRunner(AbstractRunner):
#    """A Runner uses an agent to run through an environment.
#    The experiences of the agent are stored in an external queue.
#    """
#    def __init__(self, *args):
#        
#        super(BaseRunner, self).__init__(*args)
#        
#        if map_game_to_net(self._game) == 'dense-net':
#            base_net = base_nets.DenseBaseNet()
#            
#        elif map_game_to_net(self._game) == 'conv-net':
#            base_net = base_nets.ConvBaseNet()
#            
#        
#        self._Agent = get_Agent(self._algo, 'RunnerAgent')
#        self.agent = self._Agent(base_net, **self._kwargs_agent)
#        
#        self.env = gym.make(self._game)
#
#    def run(self):
#        pass
#    
#    def train(self, coord):
#        pass
#    
#    
#
#    def thread(self, coord, mode):
#        return Thread(target=self.run)
#        
#
#class RunnerFactory:
#    def __init__(self, AgentType, num_runners=None):
#        pass
