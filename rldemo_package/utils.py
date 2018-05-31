#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:29:39 2018

@author: timo
"""

import argparse

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




