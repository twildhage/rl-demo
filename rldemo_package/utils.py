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
import argparse
import importlib
import json
import logging
import logging.config as logging_config
import os


# Third party modules 

# Custom modules
from config import Config



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
    
    argparser.add_argument('--algo', '-a', type=str, default='a3c', 
                        choices=['a3c'],
                        help='Reinforcement learning algorithm to be used.')
    
    argparser.add_argument('--cptdir', default='../../checkpoints', dest='checkpoint_dir',
                           metavar='CHECKPOINTDIR',
                        help='Set path of checkpoints.')
    
    argparser.add_argument('--device', '-d', default='cpu',
                           choices=['cpu', 'gpu'],
                        help='Set device used for training a evaluation.\
                        (Not yet supported)')
    
    argparser.add_argument('--episodes', '-e', type=int, default=Config.NUM_EPISODES,
                        help='Set number of episodes used during training. Note that in \
                        case of an experience replay buffer is used, the agent will be trained on\
                        more than the given number of episodes.')
    
    argparser.add_argument('--game', '-g', default='SpaceInvaders-v0',
                           choices=['Pendulum-v0',
                                    'Asterix-ram-v0',
                                    'CartPole-v0',
                                    'CrazyClimber-ram-v0',
                                    'TimePilot-v0',
                                    'TimePilot-v0',
                                    'Asteroids-v0',
                                    'KungFuMaster-v0',
                                    'Alien-v0',
                                    'Berzerk-v0',
                                    'MontezumaRevenge-v0',
                                    'Zaxxon-v0',
                                    'Venture-v0',
                                    'Frostbite-v0',
                                    'Seaquest-v0',
                                    'Pitfall-v0',
                                    'ElevatorAction-v0',
                                    'FishingDerby-v0',
                                    'Robotank-v0',
                                    'Jamesbond-v0',
                                    'PrivateEye-v0',
                                    'StarGunner-v0',
                                    'YarsRevenge-v0',
                                    'Boxing-v0',
                                    'Solaris-v0',
                                    'BattleZone-v0',
                                    'Gravitar-v0',
                                    'RoadRunner-v0',
                                    'Krull-v0',
                                    'Riverraid-v0',
                                    'ChopperCommand-v0',
                                    'IceHockey-v0',
                                    'Kangaroo-v0',
                                    'Freeway-v0',
                                    'Atlantis-v0',
                                    'DemonAttack-v0',
                                    'NameThisGame-v0',
                                    'Bowling-v0',
                                    'Qbert-v0',
                                    'UpNDown-v0',
                                    'Pong-v0',
                                    'Breakout-v0',
                                    'SpaceInvaders-v0',
                                    'Phoenix-v0',
                                    'BeamRider-v0',
                                    'Asterix-v0',
                                    'CrazyClimber-v0',
                                    'Enduro-v0',
                                    'MsPacman-v0',
                                    'JourneyEscape-v0',
                                    'Amidar-v0',
                                    'WizardOfWor-v0',
                                    'DoubleDunk-v0',
                                    'Centipede-v0',
                                    'Tennis-v0',
                                    'BankHeist-v0',
                                    'Skiing-v0',
                                    'Carnival-v0',
                                    'Pooyan-v0',
                                    'AirRaid-v0',
                                    'Assault-v0',
                                    'Tutankham-v0',
                                    'Gopher-v0',
                                    'VideoPinball-v0'
                                    ],
                        help='Game to be used (correspondes to OpenAI gym environment).\
                        Only a small selection of games are available in this demo.')
    
    argparser.add_argument('--logging', '-l', action='store_true', default=False,
                        help='Activate logging to file in dir /logs.')
    
    argparser.add_argument('--mode', '-m', type=str, default='train',
                        choices=['train', 'eval', 'run'],
                        help='Mode to run the RL demo.')
    
    argparser.add_argument('--print_settings', action='store_true', default=False, 
                        help='Print current settings of all commandline arguments.')
    
    argparser.add_argument('--render', action='store_true', default=False, 
                        help='Render game play.')
    
    argparser.add_argument('--share_weights', '-sw', action='store_true', default=False, 
                        help='Share weights between nets. Sharing details depend on the\
                        used reinforcement algorithm. For instance, in case of A3C\
                        weights are shared between the policy- and the value-network.')
    
    argparser.add_argument('--timesteps', '-t', type=int, default=Config.NUM_TIMESTEPS,
                        help='Set number of timesteps per episodes used during training.')
    
    argparser.add_argument('--tensorboard', '-tb', action='store_true', default=False,
                        help='Activate tensorboard summaries. (Not yet supported)')
    
    argparser.add_argument('--threads', type=int, default=1,
                        help='Set number of threads. Note: only has effect for parallel \
                        algorithms like A3C.')
    
    
    argparser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='Print live infos to the command line')
    
    args = argparser.parse_args()
    
    Config.set_cls_var('NUM_EPISODES', args.episodes)
    
    return (argparser, args)


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
            



