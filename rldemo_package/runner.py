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
from utils import get_Agent, map_game_to_net


class AbstractRunner(ABC):

    @abc.abstractmethod
    def __init__(self, algo, game, kwargs_agent):
        
        self._algo = algo
        
        self._game = game
        
        self._kwargs_agent = kwargs_agent

    @abc.abstractmethod
    def run(self, num_episodes=199, num_timesteps=50):
        pass

    @property
    @abc.abstractmethod
    def thread(self):
        pass


class BaseRunner(AbstractRunner):
    """A Runner uses an agent to run through an environment.
    The experiences of the agent are stored in an external queue.
    """
    def __init__(self, *args):
        
        super(BaseRunner, self).__init__(*args)
        
        if map_game_to_net(self._game) == 'dense-net':
            base_net = base_nets.DenseBaseNet()
            
        elif map_game_to_net(self._game) == 'conv-net':
            base_net = base_nets.ConvBaseNet()
            
        
        self._Agent = get_Agent(self._algo, 'RunnerAgent')
        self.agent = self._Agent(base_net, **self._kwargs_agent)
        
        self.env = gym.make(self._game)

    def run(self):
        pass

    def thread(self):
        return Thread(target=self.run)
        

class RunnerFactory:
    def __init__(self, AgentType, num_runners=None):
        pass


