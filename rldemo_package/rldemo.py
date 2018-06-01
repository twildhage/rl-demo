
# Build-in modules
import argparse
import logging


# Third party modules 
import tensorflow as tf
import tensorflow.contrib.eager as tfe


# Custum modules
from utils import get_argparser, print_args
from utils import get_logger, get_Manager

def init():
    argparser = get_argparser()
    args = argparser.parse_args()
    log = get_logger(args.mode, args.logging)
    print_args(args, logger=log)

    return args, log

def main():
    """Demonstration of reinforcement learning algorithms with selected OpenAI
    gym environments."""
    args, log = init()
    kwargs_agent = {'tensorboard': args.tensorboard,
                    'checkpointdir': args.checkpointdir, 
                    'share_weights': args.share_weights}
    
    Manager = get_Manager(args.algo)
    
    manager = Manager(args.mode, args.algo, args.game, kwargs_agent)
    
    manager.run(1,1)

if __name__ == '__main__':
    main()