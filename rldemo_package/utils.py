#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:29:39 2018

@author: timo
"""

# Build-in modules
import argparse
import importlib
import json
import logging
import logging.config as logging_config
import os

import pdb

# Third party modules 

# Custom modules

def get_Agent(algo, name):
    
    module_name = "algos.{algo}".format(algo=algo)
    try:
        # Dynamic module import for requested agent
        Agent = getattr(importlib.import_module(module_name), name)
    except:
        raise Exception("The algo module '{}' has to contain an Agent class.".format(algo))
    else:
        return Agent


def get_agent(algo, name, kwargs_agent):
    
    return get_Agent(algo, name)(**kwargs_agent)


def get_Manager(algo):
    
    module_name = "algos.{algo}".format(algo=algo)
    try:
        # Dynamic module import for requested manager
        Manager = getattr(importlib.import_module(module_name), 'Manager')
    except:
        raise Exception("The algo module '{}' has to contain a Manager class inherited from AbstractManager.".format(algo))
    else:
        return Manager


def get_manager(algo, kwargs_manager):
    
    return get_Manager(algo)(**kwargs_manager)


def get_argparser():
    
    
    argparser = argparse.ArgumentParser(description='Demonstration of reinforcement\
                        learning algorithms with selected OpenAI gym environments.')

#    argparser.add_argument('--discount_factor', '-df', type=float, default=config.discount_factor, 
#                        action=config(argparser, 'discount_factor', []),
#                        help='Reinforcement learning algorithm to be used.')
    
    argparser.add_argument('--algo', '-a', type=str, default='a3c', 
                        choices=['a3c', 'rainbow', 'unicorn'],
                        help='Reinforcement learning algorithm to be used.')
    
    argparser.add_argument('--cptdir', default='../../checkpoints', dest='checkpoint_dir',
                           metavar='CHECKPOINTDIR',
                        help='Set path of checkpoints.')
    
    argparser.add_argument('--device', '-d', default='cpu',
                           choices=['cpu', 'gpu'],
                        help='Set device used for training a evaluation.')
    
    argparser.add_argument('--episodes', '-e', type=int, default=100,
                        help='Set number of episodes used during training. Note that in \
                        case of an experience replay buffer is used, the agent will be trained on\
                        more than the given number of episodes.')
    
    argparser.add_argument('--game', '-g', default='Pendulum-v0',
                           choices=['Pendulum-v0',
                                    'Asterix-ram-v0',
                                    'CartPole-v0',
                                    'CrazyClimber-ram-v0'],
                        help='Game to be used (correspondes to OpenAI gym environment).\
                        Only a small selection of games are available in this demo.')
    
    argparser.add_argument('--logging', '-l', action='store_true', default=False,
                        help='Activate logging to file in dir /logs.')
    
    argparser.add_argument('--mode', '-m', type=str, default='train',
                        choices=['train', 'eval', 'run'],
                        help='Mode to run the RL demo.')
    
    argparser.add_argument('--print_settings', action='store_true', default=False, 
                        help='Print current settings of all commandline arguments.')
    
    argparser.add_argument('--share_weights', '-sw', action='store_true', default=False, 
                        help='Share weights between nets. Sharing details depend on the\
                        used reinforcement algorithm. For instance, in case of A3C\
                        weights are shared between the policy- and the value-network.')
    
    argparser.add_argument('--timesteps', '-t', type=int, default=50,
                        help='Set number of timesteps per episodes used during training.')
    
    argparser.add_argument('--tensorboard', '-tb', action='store_true', default=False,
                        help='Activate tensorboard summaries and graph analysis.')
    
    argparser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='Print live infos to the command line')

    return argparser


def map_game_to_net(game):
    
    dnn = 'dense-net'
    
    cnn = 'conv-net'
    
    _map = {'Pendulum-v0': dnn,
            'CartPole-v0': dnn,
            'Breakout': cnn}
    
    try:
        net = _map[game]
    
    except:
        raise Exception("'{game}' not available in this demo.".format(game=game))
    
    else:
        return net


def print_args(args, logger=None):
    """Convenience function to show the setting of all available commandline 
    arguments """
    
    if args.print_settings:
        for arg in vars(args):
            if logger:
                logger.info("{0}: {1}".format(arg, getattr(args, arg)))
            else:
                print("{0}: {1}".format(arg, getattr(args, arg)))


class Logger:
    """Get preconfigured logger for training and evaluation logs.
    Args:
        mode: ['train', 'eval'] The difference when chosing a mode is, that the
                logging format differs for the different modes.
        log_to_file: when False output to console only, when True also output 
                to file that correspondes to the current mode.
    """
    
    log_to_file = None
    
    mode = None

    @classmethod
    def get_logger(cls, mode, log_to_file=False):
    
        cls.mode = mode
        
        cls.log_to_file = log_to_file
        
        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        
        path = os.path.join(path_to_this_file, "../log_cfg.json")
        
        with open(path) as log_config_file:
            log_config = json.loads(log_config_file.read())
            
        logging_config.dictConfig(log_config)
        
        logger_name = 'logger_'+mode if log_to_file==False else 'logger_'+mode+'_to_file' 
        
        _logger = logging.getLogger(logger_name)
            
        return _logger

    @classmethod
    def get_active_logger(cls):
        
        try:
            logger = cls.get_logger(cls.mode, cls.log_to_file)
        
        except:
            raise Exception("No active logger. First call 'get_logger' to activate logging.")
        
        else:
            return logger
            



