#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:50:01 2018

@author: timo
"""

# Build-in modules
from queue import Queue
from threading import Thread

# Third party modules 
import gym
from tensorflow import keras
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Custom modules
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

    def train(self, coord, num_episodes=199, num_timesteps=50):
        for _ in range(num_episodes):
            episode = self.Q_experience.get()
            self.agent.train(episode)
        coord.request_stop()

    def eval(self, coord=None, num_episodes=199, num_timesteps=50):
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

    def train(self, coord=None, num_episodes=199, num_timesteps=50):
        return self.eval(coord, num_episodes, num_timesteps)

    def eval(self, coord=None, num_episodes=199, num_timesteps=50):
        
        if coord:
            
            while not coord.should_stop():
                episode = self.agent.eval(num_timesteps)
                self.Q_experience.put(episode)
        else:
            
            for _ in range(num_episodes):
                self.agent.eval(num_timesteps)


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
            
        self.presenter = Runner(self.Q_experiences, 
                                self.trainer.agent,
                                self._algo,
                                self._game,
                                self._kwargs_agent)


    def run(self, num_episodes, num_timesteps):
        
        coord = tf.train.Coordinator()
        mode = self._mode
        
        if mode == 'train':
            
            # Get threads - the trainer stops all other threads after 'num_epiodes'
            trainer_thread = self.trainer.thread(coord, mode, num_episodes=num_episodes)
            
            runner_threads = [runner.thread(coord, mode) for runner in self.runners]
            
            presenter_thread = self.presenter.thread(coord, mode)
            
            # Start threads
            trainer_thread.start()
            
            for runner_thread in runner_threads:
                runner_thread.start()
            
            presenter_thread.start()
            
            self.log.info('all threads started')
            
            # Join threads
            trainer_thread.join()
            
            for runner_thread in runner_threads:
                runner_thread.join()
            
            presenter_thread.join()
            
            self.log.info('all threads finished')
            
        elif mode == 'eval':
            self.presenter.eval(num_episodes=1,
                                num_timesteps=num_timesteps)

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
    
    def eval(self, num_timesteps):
        pass


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

    def train(self, episode):
        pass
    
    def loss_op(self, target):
        pass


