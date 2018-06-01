#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:50:01 2018

@author: timo
"""

# Build-in modules
from queue import Queue

# Third party modules 
import gym
from tensorflow import keras
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Custum modules
from base_nets import DenseBaseNet, ConvBaseNet
from manager import AbstractManager
from runner import AbstractRunner
from trainer import AbstractTrainer
from utils import map_game_to_net

import pdb

class Trainer(AbstractTrainer):

    def __init__(self, Q_experience, *args):
        
        super(Trainer, self).__init__(*args)
        
        if map_game_to_net(self._game) == 'dense-net':
            base_net = DenseBaseNet()
        
        elif map_game_to_net(self._game) == 'conv-net':
            base_net = ConvBaseNet()
        
#        pdb.set_trace() # TODO remove pdb trace
        self.agent = TrainerAgent(Q_experience, base_net, **self._kwargs_agent)
        
        self.env = gym.make(self._game)
        
        self.Q_experience = Q_experience


    def loss_op(self, target):
        pass

    def run(self, num_episodes, num_timesteps):
        pass

    def thread(self):
        pass


class Runner(AbstractRunner):

    def __init__(self, Q_experience, trainer_agent, *args):
        
        super(Runner, self).__init__(*args)
        
        if map_game_to_net(self._game) == 'dense-net':
            base_net = DenseBaseNet()
        
        elif map_game_to_net(self._game) == 'conv-net':
            base_net = ConvBaseNet()
        
        self.agent = RunnerAgent(base_net, trainer_agent, **self._kwargs_agent)
        
        self.env = gym.make(self._game)
        
        self.Q_experience = Q_experience


    def run(self, num_episodes, num_timesteps):
        pass

    def thread(self):
        pass


class Manager(AbstractManager):
    def __init__(self, *args):
        
        super(Manager, self).__init__(*args)
        
        # TODO: queue size has to be adjusted according to fill and remove rates
        self.Q_experiences = Queue(maxsize=100) 
        
        self.trainer = Trainer(self.Q_experiences, 
                               self._algo,
                               self._game,
                               self._kwargs_agent)
        
        self.num_runners = 10
        
        self.runners = []
        
        for _ in range(self.num_runners):
            self.runners.append(Runner(self.Q_experiences, 
                                       self.trainer.agent,
                                       self._algo,
                                       self._game,
                                       self._kwargs_agent))


    def run(self, num_episodes, num_timesteps):
        pass

    def summary(self):
        pass

class Policy(keras.Model):

    def __init__(self, base_net):
        
        super(Policy, self).__init__()
        
        self._base_net = base_net
        
        self._layer_out = keras.layers.Dense(2)

    def call(self, input):
        
        out = self._base_net(input)
        
        out = self._layer_out(out)
        
        return out


class ValueFunction(keras.Model):

    def __init__(self, base_net):
        
        super(ValueFunction, self).__init__()
        
        self._base_net = base_net
        
        self._layer_out = keras.layers.Dense(1)
        
        
    def call(self, input):
        
        out = self._base_net(input)
        
        out = self._layer_out(out)
        
        return out


class RunnerAgent(keras.Model):

    def __init__(self, 
                 base_net,
                 trainer_agent,
                 tensorboard,
                 checkpointdir,
                 share_weights,
                 name='A3CAgent'):
        
        super(RunnerAgent, self).__init__(name=name)
        
        self._BaseNet = base_net.__class__
        
        self._base_net = base_net 
        
        if share_weights:
            self._policy = Policy(self._base_net) 
            
            self._valuef = ValueFunction(self._base_net) 
        
        else:
            self._policy = Policy(self._BaseNet())
            
            self._valuef = ValueFunction(self._BaseNet())
            
        self.trainer_agent = trainer_agent
        
        self.share_weights = share_weights

    def policy_op(self, state):
        
        return self._policy(state)

    def valuef_op(self, state):
        
        return self._valuef(state)


class TrainerAgent(keras.Model):
    
    def __init__(self, 
                 Q_experience,
                 base_net,
                 tensorboard,
                 checkpointdir,
                 share_weights,
                 name='A3CAgent'):
        
        super(TrainerAgent, self).__init__(name=name)
        
        self._BaseNet = base_net.__class__
        
        self._base_net = base_net 
        
        if share_weights:
            self._policy = Policy(self._base_net) 
            
            self._valuef = ValueFunction(self._base_net) 
        
        else:
            self._policy = Policy(self._BaseNet())
            
            self._valuef = ValueFunction(self._BaseNet())
        
        self.share_weights = share_weights

