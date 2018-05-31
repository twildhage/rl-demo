#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:29:39 2018

@author: timo
"""

# Build-in modules
import argparse
import json
import logging
import  logging.config as logging_config
import os
from pathlib import Path
import sys

import pdb

def get_argparser():
    argparser = argparse.ArgumentParser(description='Demonstration of reinforcement\
                        learning algorithms with selected OpenAI gym environments.')
    
    argparser.add_argument('--mode', '-m', type=str, default='train',
                        choices=['train', 'eval'],
                        help='Mode to run the RL demo.')
    
    argparser.add_argument('--algo', '-a', type=str, default='a3c', 
                        choices=['a3c', 'rainbow', 'unicorn'],
                        help='Reinforcement learning algorithm to be used.')
    
    argparser.add_argument('--logging', '-l', action='store_true', default=False,
                        help='Activate logging to file in dir /logs.')
    
    argparser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='Print live infos to the command line')
    
    argparser.add_argument('--tensorboard', '-tb', action='store_true', default=False,
                        help='Activate tensorboard summaries and graph analysis.')
    
    argparser.add_argument('--game', '-g', default='Pendulum',
                           choices=['Pendulum', 'SpaceInvaders', 'HillCar'],
                        help='Game to be used (correspondes to OpenAI gym environment).\
                        Only a small selection of games are available in this demo.')
    
    argparser.add_argument('--cptdir', default='./checkpoints', dest='checkpointdir',
                           metavar='CHECKPOINTDIR',
                        help='Set path of checkpoints.')
    
    argparser.add_argument('--episodes', '-e', type=int, default=100,
                        help='Set number of episodes used during training. Note that in \
                        case of an experience replay buffer is used, the agent will be trained on\
                        more than the given number of episodes.')
    
    argparser.add_argument('--timesteps', '-t', type=int, default=50,
                        help='Set number of timesteps per episodes used during training.')
    
    argparser.add_argument('--device', '-d', default='cpu',
                           choices=['cpu', 'gpu'],
                        help='Set device used for training a evaluation.')
    
    argparser.add_argument('--print_settings', action='store_true', default=False, 
                        help='Print current settings of all commandline arguments.')
    
    return argparser


def print_args(args, logger=None):
    """Convenience function to show the setting of all available commandline 
    arguments """
    
    if args.print_settings:
        for arg in vars(args):
            if logger:
                logger.info("{0}: {1}".format(arg, getattr(args, arg)))
            else:
                print("{0}: {1}".format(arg, getattr(args, arg)))


def get_logger(mode, log_to_file=False):
    """Get preconfigured logger for training and evaluation logs.
    Args:
        mode: ['train', 'eval'] The difference when chosing a mode is, that the
                logging format differs for the different modes.
        log_to_file: when False output to console only, when True also output 
                to file that correspondes to the current mode.
    """
    
    logger_name = 'logger_'+mode if log_to_file==False else 'logger_'+mode+'_to_file' 
    
    path_to_this_file = os.path.abspath(os.path.dirname(__file__))

    path = os.path.join(path_to_this_file, "../log_cfg.json")

    with open(path) as log_config_file:
        log_config = json.loads(log_config_file.read())

    logging_config.dictConfig(log_config)

    _logger = logging.getLogger(logger_name)
        
    return _logger






