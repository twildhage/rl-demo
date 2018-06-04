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
import logging


# Third party modules 
import tensorflow as tf
import tensorflow.contrib.eager as tfe


# Custom modules
from utils import get_argparser, print_args
from utils import Logger, get_Manager

def init():
    
    tfe.enable_eager_execution()
    
    argparser, args = get_argparser()
    
    log = Logger.get_logger(args.mode, args.logging)
    
    print_args(args, logger=log)

    return args, log

def main():
    """Demonstration of reinforcement learning algorithms with selected OpenAI
    gym environments."""
    
    args, log = init()
    
    kwargs_agent = {'tensorboard': args.tensorboard,
                    'checkpoint_dir': args.checkpoint_dir, 
                    'share_weights': args.share_weights}
    
    Manager = get_Manager(args.algo)
    
    manager = Manager(args.mode, args.algo, args.game, kwargs_agent)
    
    num_runners = manager.num_runners if args.mode=='train' else 1
    
    manager.run(args.episodes*num_runners, args.timesteps)

if __name__ == '__main__':
    main()